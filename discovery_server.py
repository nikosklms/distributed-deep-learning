#!/usr/bin/env python3
# discovery_server.py
# Discovery server for fault-tolerant distributed training
# Uses initial discovery phase with timer for robustness

import socket
import threading
import time
import pickle
import struct
from collections import defaultdict

# Configuration
DISCOVERY_PORT = 5000
INITIAL_DISCOVERY_TIME = 20  # Wait up to 20s for all nodes
EXPECTED_NODES = None  # Auto-detect from first registration

class DiscoveryServer:
    def __init__(self, port=DISCOVERY_PORT):
        self.port = port
        self.registrations = defaultdict(dict)  # world_size -> {rank: node_info}
        self.discovery_complete = defaultdict(threading.Event)  # world_size -> Event
        self.lock = threading.Lock()
        
    def recv_all(self, sock, n):
        """Helper to receive exactly n bytes"""
        data = bytearray()
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)
    
    def handle_registration(self, conn, addr):
        try:
            # Receive registration message
            raw_len = self.recv_all(conn, 4)
            if not raw_len:
                return
            
            msg_len = struct.unpack('>I', raw_len)[0]
            data = self.recv_all(conn, msg_len)
            if not data:
                return
            
            msg = pickle.loads(data)
            
            rank = msg['rank']
            world_size = msg['world_size']
            data_port = msg.get('data_port', 0)
            udp_port = msg.get('udp_port', 0)
            
            # Register node
            with self.lock:
                self.registrations[world_size][rank] = {
                    'rank': rank,
                    'ip': addr[0],
                    'data_port': data_port,
                    'udp_port': udp_port
                }
                num_registered = len(self.registrations[world_size])
            
            print(f"[Discovery] Registered rank {rank}/{world_size} from {addr[0]} ({num_registered}/{world_size})")
            
            # Start discovery timer for this world_size if first node
            if num_registered == 1:
                threading.Thread(
                    target=self.discovery_timer,
                    args=(world_size,),
                    daemon=True
                ).start()
            
            # Check if all nodes registered
            if num_registered == world_size:
                print(f"[Discovery] All {world_size} nodes registered! Discovery complete.")
                self.discovery_complete[world_size].set()
            
            # Wait for discovery to complete
            self.discovery_complete[world_size].wait()
            
            # Send complete node list
            with self.lock:
                node_list = self.registrations[world_size]
            
            response = pickle.dumps(node_list)
            conn.sendall(struct.pack('>I', len(response)) + response)
            
            print(f"[Discovery] Sent node list to rank {rank}/{world_size}")
            
        except Exception as e:
            print(f"[Discovery] Error handling {addr}: {e}")
        finally:
            conn.close()
    
    def discovery_timer(self, world_size):
        """Wait for INITIAL_DISCOVERY_TIME, then mark discovery complete"""
        print(f"[Discovery] Timer started for world_size={world_size}: waiting {INITIAL_DISCOVERY_TIME}s...")
        
        start_time = time.time()
        while time.time() - start_time < INITIAL_DISCOVERY_TIME:
            with self.lock:
                if len(self.registrations[world_size]) >= world_size:
                    # All nodes arrived early!
                    return
            time.sleep(0.5)
        
        # Timer expired
        with self.lock:
            num_registered = len(self.registrations[world_size])
        
        print(f"[Discovery] Timer expired. Registered: {num_registered}/{world_size}")
        print(f"[Discovery] Discovery complete for world_size={world_size}")
        
        self.discovery_complete[world_size].set()
    
    def run(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(('0.0.0.0', self.port))
        server_sock.listen(10)
        
        print(f"[Discovery] Server listening on port {self.port}")
        
        try:
            while True:
                conn, addr = server_sock.accept()
                threading.Thread(
                    target=self.handle_registration,
                    args=(conn, addr),
                    daemon=True
                ).start()
        except KeyboardInterrupt:
            print("\n[Discovery] Shutting down...")
        finally:
            server_sock.close()


if __name__ == "__main__":
    server = DiscoveryServer()
    server.run()