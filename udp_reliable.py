#!/usr/bin/env python3
# udp_reliable.py
# Reliable UDP messaging with ACK/retransmission for fault-tolerant distributed training

import socket
import threading
import time
import struct
import pickle
import random
from collections import defaultdict
from enum import IntEnum

class MessageType(IntEnum):
    """UDP Message types for fault tolerance protocol"""
    HEARTBEAT = 1
    FAILURE_NOTIFY = 2
    RECOVERY_NOTIFY = 3
    ACK = 4

class ReliableUDP:
    """
    Reliable UDP messaging layer with:
    - Sequence numbers for ordering
    - Session IDs to handle node restarts (fixes duplicate drop issue)
    - ACK mechanism for reliability
    - Retransmission with exponential backoff
    - Deduplication
    """
    
    def __init__(self, rank, listen_port, verbose=False):
        """
        Initialize reliable UDP layer
        
        Args:
            rank: This node's rank
            listen_port: UDP port to listen on
            verbose: Enable debug logging
        """
        self.rank = rank
        self.listen_port = listen_port
        self.verbose = verbose
        
        # Generate a random Session ID for this run.
        # This allows peers to distinguish between a delayed packet from a previous run
        # and a new packet from a restarted node (seqnum reset).
        self.session_id = random.getrandbits(32)
        
        # Socket setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', listen_port))
        self.sock.settimeout(0.1)  # Non-blocking with timeout
        
        # Sequence number management
        self.seqnum_lock = threading.Lock()
        self.next_seqnum = 0
        
        # ACK tracking for sent messages
        self.pending_acks = {}  # seqnum -> (message, dest_addr, timestamp, retries)
        self.ack_lock = threading.Lock()
        
        # Received message deduplication
        # Map: sender_addr -> {'session_id': int, 'seen_seqnums': set()}
        self.peer_states = {}
        self.peer_state_lock = threading.Lock()
        
        # Message handlers
        self.handlers = {
            MessageType.HEARTBEAT: None,
            MessageType.FAILURE_NOTIFY: None,
            MessageType.RECOVERY_NOTIFY: None,
        }
        
        # Control
        self.stop_flag = False
        
        # Start background threads
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.retransmit_thread = threading.Thread(target=self._retransmit_loop, daemon=True)
        
        self.listener_thread.start()
        self.retransmit_thread.start()
        
        if self.verbose:
            print(f"[UDP {self.rank}] Listening on port {listen_port} (Session ID: {self.session_id})")
    
    def _get_next_seqnum(self):
        """Get next sequence number (thread-safe)"""
        with self.seqnum_lock:
            seqnum = self.next_seqnum
            self.next_seqnum += 1
            return seqnum
    
    def _pack_message(self, msg_type, seqnum, payload):
        """
        Pack message into wire format
        
        Format: [msg_type(1B)][seqnum(4B)][session_id(4B)][payload_len(4B)][payload]
        """
        payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        # Added session_id to header
        header = struct.pack('>BIII', msg_type, seqnum, self.session_id, len(payload_bytes))
        return header + payload_bytes
    
    def _unpack_message(self, data):
        """
        Unpack message from wire format
        
        Returns: (msg_type, seqnum, session_id, payload) or None if invalid
        """
        try:
            # Header size is now 1 + 4 + 4 + 4 = 13 bytes
            header_size = 13
            if len(data) < header_size:  # Minimum header size
                return None
            
            msg_type, seqnum, session_id, payload_len = struct.unpack('>BIII', data[:header_size])
            
            if len(data) < header_size + payload_len:
                return None
            
            payload_bytes = data[header_size:header_size+payload_len]
            payload = pickle.loads(payload_bytes)
            
            return (MessageType(msg_type), seqnum, session_id, payload)
        except Exception as e:
            if self.verbose:
                print(f"[UDP {self.rank}] Failed to unpack message: {e}")
            return None
    
    def _is_duplicate_and_update(self, sender_addr, session_id, seqnum):
        """
        Check if duplicate AND update state for sender.
        Handles restarts by checking session_id.
        """
        with self.peer_state_lock:
            # If new peer or peer restarted (new session ID)
            if sender_addr not in self.peer_states or self.peer_states[sender_addr]['session_id'] != session_id:
                if self.verbose and sender_addr in self.peer_states:
                    print(f"[UDP {self.rank}] Peer {sender_addr} restarted (New Session: {session_id}). Resetting seqnums.")
                
                # Reset state for this peer
                self.peer_states[sender_addr] = {
                    'session_id': session_id,
                    'seen_seqnums': {seqnum} # Add current seqnum
                }
                
                # Trim overall peer list if too large (prevent memory leak from random IPs)
                if len(self.peer_states) > 1000:
                     # Remove arbitrary key
                    self.peer_states.pop(next(iter(self.peer_states)))
                
                return False

            # Known peer, same session
            state = self.peer_states[sender_addr]
            if seqnum in state['seen_seqnums']:
                return True
            
            # Add to seen set
            state['seen_seqnums'].add(seqnum)
            
            # Keep window size reasonable
            if len(state['seen_seqnums']) > 1000:
                # Remove oldest (approximate via min)
                state['seen_seqnums'].remove(min(state['seen_seqnums']))
            
            return False
    
    def _send_ack(self, dest_addr, seqnum):
        """Send ACK for a received message"""
        ack_msg = self._pack_message(MessageType.ACK, seqnum, {})
        try:
            self.sock.sendto(ack_msg, dest_addr)
            # Verbose ACK logging disabled to reduce noise, enable if debugging specific ACK issues
            # if self.verbose:
            #     print(f"[UDP {self.rank}] Sent ACK for seqnum {seqnum} to {dest_addr}")
        except Exception as e:
            if self.verbose:
                print(f"[UDP {self.rank}] Failed to send ACK: {e}")
    
    def _listen_loop(self):
        """Background thread: receive and process incoming UDP messages"""
        while not self.stop_flag:
            try:
                data, addr = self.sock.recvfrom(65535)
                
                result = self._unpack_message(data)
                if not result:
                    continue
                
                msg_type, seqnum, session_id, payload = result
                
                # Handle ACK messages
                if msg_type == MessageType.ACK:
                    with self.ack_lock:
                        if seqnum in self.pending_acks:
                            del self.pending_acks[seqnum]
                            if self.verbose:
                                print(f"[UDP {self.rank}] Received ACK for seqnum {seqnum}")
                    continue
                
                # Check for duplicates AND handle restarts via session_id
                if self._is_duplicate_and_update(addr, session_id, seqnum):
                    if self.verbose:
                        print(f"[UDP {self.rank}] Duplicate message (seqnum {seqnum}) from {addr}, ignoring")
                    # Still send ACK (in case our previous ACK was lost)
                    self._send_ack(addr, seqnum)
                    continue
                
                # Send ACK for this message
                self._send_ack(addr, seqnum)
                
                # Dispatch to handler
                if msg_type in self.handlers and self.handlers[msg_type]:
                    try:
                        self.handlers[msg_type](payload, addr)
                    except Exception as e:
                        if self.verbose:
                            print(f"[UDP {self.rank}] Handler error for {msg_type.name}: {e}")
                
            except socket.timeout:
                continue
            except Exception as e:
                if not self.stop_flag and self.verbose:
                    print(f"[UDP {self.rank}] Listen loop error: {e}")
    
    def _retransmit_loop(self):
        """Background thread: retransmit messages that haven't been ACKed"""
        while not self.stop_flag:
            time.sleep(0.5)  # Check every 500ms
            
            current_time = time.time()
            to_retransmit = []
            
            with self.ack_lock:
                for seqnum, (message, dest_addr, timestamp, retries) in list(self.pending_acks.items()):
                    # Exponential backoff: 0.5s, 1s, 2s, 4s...
                    timeout = min(0.5 * (2 ** retries), 4.0)
                    
                    if current_time - timestamp > timeout:
                        if retries < 10:  # Max 10 retries (more aggressive for recovery)
                            to_retransmit.append((seqnum, message, dest_addr, retries))
                        else:
                            # Give up after retries
                            if self.verbose:
                                print(f"[UDP {self.rank}] Gave up on seqnum {seqnum} after max retries")
                            del self.pending_acks[seqnum]
            
            # Retransmit outside the lock
            for seqnum, message, dest_addr, retries in to_retransmit:
                try:
                    self.sock.sendto(message, dest_addr)
                    if self.verbose:
                        print(f"[UDP {self.rank}] Retransmitting seqnum {seqnum} (retry {retries+1})")
                    
                    with self.ack_lock:
                        if seqnum in self.pending_acks:
                            self.pending_acks[seqnum] = (message, dest_addr, current_time, retries + 1)
                except Exception as e:
                    if self.verbose:
                        print(f"[UDP {self.rank}] Retransmit error: {e}")
    
    def send_reliable(self, msg_type, payload, dest_addr):
        """
        Send a message reliably (with ACK/retransmit)
        
        Args:
            msg_type: MessageType enum value
            payload: Dictionary payload
            dest_addr: (ip, port) tuple
        """
        seqnum = self._get_next_seqnum()
        message = self._pack_message(msg_type, seqnum, payload)
        
        # Add to pending ACKs
        with self.ack_lock:
            self.pending_acks[seqnum] = (message, dest_addr, time.time(), 0)
        
        # Send initial transmission
        try:
            self.sock.sendto(message, dest_addr)
            if self.verbose:
                print(f"[UDP {self.rank}] Sent {msg_type.name} (seqnum {seqnum}) to {dest_addr}")
        except Exception as e:
            if self.verbose:
                print(f"[UDP {self.rank}] Send error: {e}")
    
    def send_unreliable(self, msg_type, payload, dest_addr):
        """
        Send a message without ACK/retransmit (for heartbeats)
        
        Args:
            msg_type: MessageType enum value
            payload: Dictionary payload
            dest_addr: (ip, port) tuple
        """
        seqnum = self._get_next_seqnum()
        message = self._pack_message(msg_type, seqnum, payload)
        
        try:
            self.sock.sendto(message, dest_addr)
        except Exception as e:
            if self.verbose:
                print(f"[UDP {self.rank}] Send unreliable error: {e}")
    
    def register_handler(self, msg_type, handler_func):
        """
        Register a callback for a message type
        
        Args:
            msg_type: MessageType enum value
            handler_func: Function(payload, sender_addr) to call when message received
        """
        self.handlers[msg_type] = handler_func
    
    def close(self):
        """Clean shutdown"""
        self.stop_flag = True
        
        # Wait for threads
        if self.listener_thread.is_alive():
            self.listener_thread.join(timeout=2.0)
        if self.retransmit_thread.is_alive():
            self.retransmit_thread.join(timeout=2.0)
        
        # Close socket
        try:
            self.sock.close()
        except:
            pass
        
        if self.verbose:
            print(f"[UDP {self.rank}] Closed")


# ============================================================================
# TESTING CODE
# ============================================================================

def test_reliable_udp():
    """Test the reliable UDP layer with two nodes"""
    print("\n" + "="*60)
    print("Testing Reliable UDP Layer")
    print("="*60 + "\n")
    
    # Create two UDP endpoints
    node0 = ReliableUDP(rank=0, listen_port=7000, verbose=True)
    node1 = ReliableUDP(rank=1, listen_port=7001, verbose=True)
    
    # Message counters
    node0_received = []
    node1_received = []
    
    # Register handlers
    def node0_handler(payload, addr):
        print(f"[Node 0 Handler] Received from {addr}: {payload}")
        node0_received.append(payload)
    
    def node1_handler(payload, addr):
        print(f"[Node 1 Handler] Received from {addr}: {payload}")
        node1_received.append(payload)
    
    node0.register_handler(MessageType.HEARTBEAT, node0_handler)
    node1.register_handler(MessageType.HEARTBEAT, node1_handler)
    
    # Test 1: Simple send/receive
    print("\n--- Test 1: Simple Send/Receive ---")
    node0.send_reliable(
        MessageType.HEARTBEAT,
        {'test': 'hello from node 0'},
        ('127.0.0.1', 7001)
    )
    time.sleep(2)
    assert len(node1_received) == 1, "Node 1 should receive 1 message"
    print("Test 1 passed: Message delivered and ACKed\n")
    
    # Test 2: Restart Simulation (Session ID check)
    print("\n--- Test 2: Node Restart (Session ID) ---")
    
    # Simulate Node 0 restarting (new instance, new session_id, seqnum resets to 0)
    node0.close()
    print("Node 0 restarting...")
    node0_new = ReliableUDP(rank=0, listen_port=7000, verbose=True)
    
    # Node 0 sends a message (Seqnum 0)
    node0_new.send_reliable(
        MessageType.HEARTBEAT,
        {'test': 'I have restarted!'},
        ('127.0.0.1', 7001)
    )
    time.sleep(2)
    
    # Node 1 should accept this, even though it saw Seqnum 0 from the old Node 0
    # because the Session ID changed.
    assert len(node1_received) == 2, f"Node 1 should accept restart message (got {len(node1_received)})"
    print("Test 2 passed: Restarted node message accepted (duplicate check handled via Session ID)\n")
    
    # Cleanup
    print("\n--- Cleanup ---")
    node0_new.close()
    node1.close()
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_reliable_udp()