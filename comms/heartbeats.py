#!/usr/bin/env python3
# heartbeat.py
# Heartbeat system for fault detection in distributed training

import threading
import time
from comms.udp_reliable import ReliableUDP, MessageType

class HeartbeatMonitor:
    """
    Heartbeat-based failure detection system
    
    Each node:
    - Sends periodic heartbeats to its RIGHT neighbor
    - Monitors heartbeats from its LEFT neighbor
    - Detects failure if no heartbeat received within timeout
    """
    
    def __init__(self, rank, world_size, udp_layer, heartbeat_interval=2.0, heartbeat_timeout=10.0, verbose=False):
        """
        Initialize heartbeat monitor
        
        Args:
            rank: This node's rank
            world_size: Total number of nodes
            udp_layer: ReliableUDP instance
            heartbeat_interval: Seconds between heartbeats (default 2s)
            heartbeat_timeout: Seconds before declaring failure (default 10s)
            verbose: Enable debug logging
        """
        self.rank = rank
        self.world_size = world_size
        self.udp = udp_layer
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.verbose = verbose
        
        # Neighbor information (set via set_neighbors)
        self.left_neighbor = None   # Who sends data TO me (who I monitor)
        self.right_neighbor = None  # Who I send data TO (who I heartbeat)
        
        # Heartbeat state
        self.heartbeat_lock = threading.Lock()
        self.last_heartbeat_recv = None  # Timestamp of last received heartbeat
        self.neighbor_alive = True       # Is left neighbor alive?
        
        # Failure detection callback
        self.failure_callback = None  # Called when failure detected
        
        # Control
        self.stop_flag = False
        self.paused = False  # Can pause heartbeat sending (e.g., during recovery)
        
        # Register handler for incoming heartbeats
        self.udp.register_handler(MessageType.HEARTBEAT, self._handle_heartbeat)
        
        # Start background threads
        self.sender_thread = threading.Thread(target=self._heartbeat_sender_loop, daemon=True)
        self.monitor_thread = threading.Thread(target=self._heartbeat_monitor_loop, daemon=True)
        
        self.sender_thread.start()
        self.monitor_thread.start()
        
        if self.verbose:
            print(f"[Heartbeat {self.rank}] Monitor initialized (interval={heartbeat_interval}s, timeout={heartbeat_timeout}s)")
    
    def set_neighbors(self, left_neighbor_info, right_neighbor_info):
        """
        Set neighbor information after ring setup
        
        Args:
            left_neighbor_info: Dict with 'rank', 'ip', 'udp_port'
            right_neighbor_info: Dict with 'rank', 'ip', 'udp_port'
        """
        self.left_neighbor = left_neighbor_info
        self.right_neighbor = right_neighbor_info
        
        # Start monitoring immediately
        with self.heartbeat_lock:
            self.last_heartbeat_recv = time.time()
            self.neighbor_alive = True
        
        if self.verbose:
            print(f"[Heartbeat {self.rank}] Monitoring left neighbor: rank {left_neighbor_info['rank']}")
            print(f"[Heartbeat {self.rank}] Sending heartbeats to right neighbor: rank {right_neighbor_info['rank']}")
    
    def set_failure_callback(self, callback):
        """
        Register callback for failure detection
        
        Args:
            callback: Function(failed_rank) called when failure detected
        """
        self.failure_callback = callback
    
    def pause_heartbeats(self):
        """Pause sending heartbeats (used during recovery/handshake)"""
        self.paused = True
        if self.verbose:
            print(f"[Heartbeat {self.rank}] Paused")
    
    def resume_heartbeats(self):
        """Resume sending heartbeats"""
        self.paused = False
        # Reset monitoring state
        with self.heartbeat_lock:
            self.last_heartbeat_recv = time.time()
            self.neighbor_alive = True
        if self.verbose:
            print(f"[Heartbeat {self.rank}] Resumed")
    
    def _handle_heartbeat(self, payload, sender_addr):
        """
        Handle incoming heartbeat from left neighbor
        
        Args:
            payload: Dict with heartbeat data
            sender_addr: (ip, port) of sender
        """
        sender_rank = payload.get('from_rank', -1)
        
        # Verify it's from our left neighbor
        if self.left_neighbor and sender_rank != self.left_neighbor['rank']:
            if self.verbose:
                print(f"[Heartbeat {self.rank}] Ignoring heartbeat from unexpected rank {sender_rank}")
            return
        
        # Update last heartbeat time
        with self.heartbeat_lock:
            self.last_heartbeat_recv = time.time()
            
            # If neighbor was dead, mark as alive again
            if not self.neighbor_alive:
                self.neighbor_alive = True
                if self.verbose:
                    print(f"[Heartbeat {self.rank}] Left neighbor (rank {sender_rank}) is alive again")
        
        if self.verbose:
            print(f"[Heartbeat {self.rank}] Received heartbeat from rank {sender_rank}")
    
    def _heartbeat_sender_loop(self):
        """Background thread: send periodic heartbeats to right neighbor"""
        while not self.stop_flag:
            # Wait for neighbors to be set
            if not self.right_neighbor:
                time.sleep(0.5)
                continue
            
            # Skip if paused
            if self.paused:
                time.sleep(0.5)
                continue
            
            # Send heartbeat
            try:
                dest_addr = (self.right_neighbor['ip'], self.right_neighbor['udp_port'])
                payload = {
                    'from_rank': self.rank,
                    'timestamp': time.time()
                }
                
                # Use unreliable send (fire-and-forget) for heartbeats
                self.udp.send_unreliable(MessageType.HEARTBEAT, payload, dest_addr)
                
                if self.verbose:
                    print(f"[Heartbeat {self.rank}] Sent heartbeat to rank {self.right_neighbor['rank']}")
                
            except Exception as e:
                if self.verbose:
                    print(f"[Heartbeat {self.rank}] Error sending heartbeat: {e}")
            
            # Sleep until next heartbeat
            time.sleep(self.heartbeat_interval)
    
    def _heartbeat_monitor_loop(self):
        """Background thread: monitor heartbeats from left neighbor"""
        while not self.stop_flag:
            # Wait for neighbors to be set
            if not self.left_neighbor:
                time.sleep(0.5)
                continue
            
            # Skip monitoring if paused
            if self.paused:
                time.sleep(0.5)
                continue
            
            # Check for timeout
            with self.heartbeat_lock:
                if self.last_heartbeat_recv is None:
                    # Haven't received first heartbeat yet
                    time.sleep(0.5)
                    continue
                
                time_since_heartbeat = time.time() - self.last_heartbeat_recv
                
                if time_since_heartbeat > self.heartbeat_timeout:
                    if self.neighbor_alive:
                        # Failure detected!
                        self.neighbor_alive = False
                        failed_rank = self.left_neighbor['rank']
                        
                        print(f"\n[Heartbeat {self.rank}] FAILURE DETECTED!")
                        print(f"[Heartbeat {self.rank}] Left neighbor (rank {failed_rank}) timeout after {time_since_heartbeat:.1f}s")
                        
                        # Call failure callback
                        if self.failure_callback:
                            try:
                                self.failure_callback(failed_rank)
                            except Exception as e:
                                print(f"[Heartbeat {self.rank}] Error in failure callback: {e}")
            
            # Check every second
            time.sleep(1.0)
    
    def is_neighbor_alive(self):
        """Check if left neighbor is currently alive"""
        with self.heartbeat_lock:
            return self.neighbor_alive
    
    def reset_monitor(self):
        """Reset monitoring state (called after recovery)"""
        with self.heartbeat_lock:
            self.last_heartbeat_recv = time.time()
            self.neighbor_alive = True
        if self.verbose:
            print(f"[Heartbeat {self.rank}] Monitor reset")
    
    def close(self):
        """Clean shutdown"""
        self.stop_flag = True
        
        # Wait for threads
        if self.sender_thread.is_alive():
            self.sender_thread.join(timeout=2.0)
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        if self.verbose:
            print(f"[Heartbeat {self.rank}] Closed")


# ============================================================================
# TESTING CODE
# ============================================================================

def test_heartbeat_system():
    """Test heartbeat system with 3 nodes in a ring"""
    print("\n" + "="*60)
    print("Testing Heartbeat System")
    print("="*60 + "\n")
    
    # Create 3 nodes in a ring
    udp0 = ReliableUDP(rank=0, listen_port=8000, verbose=False)
    udp1 = ReliableUDP(rank=1, listen_port=8001, verbose=False)
    udp2 = ReliableUDP(rank=2, listen_port=8002, verbose=False)
    
    hb0 = HeartbeatMonitor(rank=0, world_size=3, udp_layer=udp0, 
                           heartbeat_interval=1.0, heartbeat_timeout=5.0, verbose=True)
    hb1 = HeartbeatMonitor(rank=1, world_size=3, udp_layer=udp1,
                           heartbeat_interval=1.0, heartbeat_timeout=5.0, verbose=True)
    hb2 = HeartbeatMonitor(rank=2, world_size=3, udp_layer=udp2,
                           heartbeat_interval=1.0, heartbeat_timeout=5.0, verbose=True)
    
    # Set up ring: 0 -> 1 -> 2 -> 0
    hb0.set_neighbors(
        left_neighbor_info={'rank': 2, 'ip': '127.0.0.1', 'udp_port': 8002},
        right_neighbor_info={'rank': 1, 'ip': '127.0.0.1', 'udp_port': 8001}
    )
    hb1.set_neighbors(
        left_neighbor_info={'rank': 0, 'ip': '127.0.0.1', 'udp_port': 8000},
        right_neighbor_info={'rank': 2, 'ip': '127.0.0.1', 'udp_port': 8002}
    )
    hb2.set_neighbors(
        left_neighbor_info={'rank': 1, 'ip': '127.0.0.1', 'udp_port': 8001},
        right_neighbor_info={'rank': 0, 'ip': '127.0.0.1', 'udp_port': 8000}
    )
    
    # Failure detection tracking
    failures_detected = []
    
    def failure_callback(failed_rank):
        failures_detected.append(failed_rank)
        print(f"\n*** FAILURE CALLBACK: Node {failed_rank} failed! ***\n")
    
    hb0.set_failure_callback(failure_callback)
    hb1.set_failure_callback(failure_callback)
    hb2.set_failure_callback(failure_callback)
    
    # Test 1: Normal operation
    print("\n--- Test 1: Normal Heartbeat Operation ---")
    time.sleep(5)
    assert hb0.is_neighbor_alive(), "Node 0 should see node 2 alive"
    assert hb1.is_neighbor_alive(), "Node 1 should see node 0 alive"
    assert hb2.is_neighbor_alive(), "Node 2 should see node 1 alive"
    print("Test 1 passed: All nodes alive\n")
    
    # Test 2: Simulate failure (pause node 1)
    print("\n--- Test 2: Simulated Failure ---")
    print("Pausing node 1 heartbeats (simulating failure)...")
    hb1.pause_heartbeats()
    
    # Wait for timeout + buffer
    time.sleep(7)
    
    # Node 2 should detect node 1 failure
    assert not hb2.is_neighbor_alive(), "Node 2 should detect node 1 failure"
    assert len(failures_detected) >= 1, "Should detect at least 1 failure"
    assert 1 in failures_detected, "Should detect node 1 failure"
    print("Test 2 passed: Failure detected correctly\n")
    
    # Test 3: Recovery
    print("\n--- Test 3: Recovery ---")
    print("Resuming node 1 heartbeats (simulating recovery)...")
    hb1.resume_heartbeats()
    hb2.reset_monitor()  # Reset after recovery
    
    time.sleep(4)
    
    # Node 2 should see node 1 alive again
    assert hb2.is_neighbor_alive(), "Node 2 should see node 1 alive after recovery"
    print("Test 3 passed: Recovery successful\n")
    
    # Test 4: Pause/resume functionality
    print("\n--- Test 4: Pause/Resume ---")
    hb0.pause_heartbeats()
    time.sleep(2)
    hb0.resume_heartbeats()
    time.sleep(2)
    assert hb1.is_neighbor_alive(), "Node 1 should still see node 0 alive"
    print("Test 4 passed: Pause/resume works\n")
    
    # Cleanup
    print("\n--- Cleanup ---")
    hb0.close()
    hb1.close()
    hb2.close()
    udp0.close()
    udp1.close()
    udp2.close()
    
    print("\n" + "="*60)
    print("All tests passed!")
    print(f"Total failures detected: {len(failures_detected)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_heartbeat_system()