#!/usr/bin/env python3
"""
allreduce_cpu.py - CPU Ring AllReduce Communication Library

This module provides a pure-Python CPU implementation of ring allreduce
for distributed gradient averaging. No GPU required.

Usage:
    from allreduce_cpu import RingAllReducer
    
    comm = RingAllReducer(rank=0, world_size=2)
    averaged = comm.allreduce(gradient_list)
    comm.close()
"""

import socket
import threading
import time
import numpy as np
import pickle
import struct
import os

# Configuration - can be overridden via environment variables
DISCOVERY_IP = os.getenv("DISCOVERY_IP", "127.0.0.1")
DISCOVERY_PORT = int(os.getenv("DISCOVERY_PORT", "5000"))
BASE_PORT = int(os.getenv("BASE_PORT", "6000"))


def _recv_all_pickle(sock):
    """Receive a pickled object with 4-byte length prefix"""
    raw_len = b''
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            return None
        raw_len += chunk
    msg_len = struct.unpack('>I', raw_len)[0]
    data = b''
    while len(data) < msg_len:
        chunk = sock.recv(msg_len - len(data))
        if not chunk:
            return None
        data += chunk
    return pickle.loads(data)


def _send_chunk(sock, chunk_list):
    """Send a list of numpy arrays as a single byte buffer with 8-byte length header"""
    all_bytes = b''.join([arr.astype(np.float32).tobytes() for arr in chunk_list])
    sock.sendall(len(all_bytes).to_bytes(8, 'big'))
    sock.sendall(all_bytes)


def _recv_bytes(sock):
    """Receive a byte buffer with 8-byte length header"""
    header = b''
    while len(header) < 8:
        chunk = sock.recv(8 - len(header))
        if not chunk:
            return None
        header += chunk

    data_length = int.from_bytes(header, 'big')
    if data_length <= 0 or data_length > 1 << 30:  # 1GB limit
        raise ValueError(f"Invalid data length: {data_length}")

    data_bytes = bytearray(data_length)
    view = memoryview(data_bytes)
    
    while data_length > 0:
        nbytes = sock.recv_into(view, data_length)
        if not nbytes:
            raise EOFError("Connection broken while receiving data")
        view = view[nbytes:]
        data_length -= nbytes

    return data_bytes


def _unpack_bytes(data_bytes, template_chunk_list):
    """Unpack raw bytes into numpy arrays based on template shapes"""
    received_chunk = []
    offset = 0
    
    for template_arr in template_chunk_list:
        num_bytes = int(np.prod(template_arr.shape)) * 4  # 4 bytes for float32
        arr_bytes = data_bytes[offset:offset + num_bytes]
        new_arr = np.frombuffer(arr_bytes, dtype=np.float32).reshape(template_arr.shape).copy()
        received_chunk.append(new_arr)
        offset += num_bytes
    
    if offset != len(data_bytes):
        raise ValueError(f"Unpack size mismatch. Expected {offset} bytes, got {len(data_bytes)}")

    return received_chunk


def _chunk_list(lst, n):
    """Split a list into n roughly equal chunks"""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


class RingAllReducer:
    """
    CPU Ring AllReduce for distributed gradient averaging.
    
    This is a pure-Python implementation that works on any machine with numpy.
    Uses TCP sockets for communication between nodes.
    
    Args:
        rank: This node's rank (0 to world_size-1)
        world_size: Total number of nodes
        verbose: Print debug messages
    
    Example:
        # On node 0:
        comm = RingAllReducer(rank=0, world_size=2)
        
        # After computing gradients:
        avg_grads = comm.allreduce([layer1.d_weights, layer1.d_biases, ...])
        
        # Apply averaged gradients
        layer1.d_weights, layer1.d_biases, ... = avg_grads
        
        # Cleanup
        comm.close()
    """
    
    def __init__(self, rank, world_size, verbose=True):
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose

        # Get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        finally:
            s.close()

        self.listen_port = BASE_PORT + self.rank
        self._log(f"Listen port: {self.listen_port}")

        # Discovery registration
        self.my_info = {
            "ip": local_ip,
            "data_port": self.listen_port,
            "rank": self.rank,
            "world_size": self.world_size
        }
        
        # Connect to discovery server
        self._log(f"Connecting to discovery server at {DISCOVERY_IP}:{DISCOVERY_PORT}...")
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_disc:
                    s_disc.connect((DISCOVERY_IP, DISCOVERY_PORT))
                    serialized = pickle.dumps(self.my_info)
                    s_disc.sendall(struct.pack('>I', len(serialized)) + serialized)
                    all_clients = _recv_all_pickle(s_disc)
                
                if not all_clients:
                    time.sleep(2)
                    continue
                
                if isinstance(all_clients, dict):
                    all_clients = list(all_clients.values())
                
                if len(all_clients) == self.world_size:
                    self._log(f"All {self.world_size} nodes discovered!")
                    break
                else:
                    self._log(f"Waiting... ({len(all_clients)}/{self.world_size} nodes)")
                    time.sleep(2)
            except Exception as e:
                self._log(f"Discovery error: {e}, retrying...")
                time.sleep(2)

        # Setup ring topology
        all_clients.sort(key=lambda x: x["rank"])
        my_index = next(i for i, c in enumerate(all_clients) if c["rank"] == self.rank)

        self.right_neighbor_info = all_clients[(my_index + 1) % world_size]
        self.left_neighbor_info = all_clients[(my_index - 1 + world_size) % world_size]
        self.connect_port = self.right_neighbor_info.get("data_port") or self.right_neighbor_info.get("port")

        self.listener_sock = None
        self.sender_sock = None

        self._log(f"Ring: rank {self.left_neighbor_info['rank']} → me → rank {self.right_neighbor_info['rank']}")

        # Internal state
        self.chunks = []
        self.buffer_full = False
        self.cond = threading.Condition()
        self.snd_mtx = threading.Lock()
        self.snd_mtx.acquire()
        self.listen_mtx = threading.Lock()
        self.listen_mtx.acquire()
        self.iteration_cnt = 0
        self.stop_flag = False
        self.listener_cnt = 0

        # Start worker threads
        self.listener = threading.Thread(target=self._listen_thread, daemon=True)
        self.listener.start()
        self.sender = threading.Thread(target=self._sender_thread, daemon=True)
        self.sender.start()
    
    def _log(self, msg):
        if self.verbose:
            print(f"[CPU Rank {self.rank}] {msg}")

    def _listen_thread(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_port = self.my_info.get("data_port")
            s.bind(('0.0.0.0', listen_port))
            s.listen()
            self._log(f"Listening on port {listen_port}...")
            conn, addr = s.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.listener_sock = conn
            self._log(f"Connected from left neighbor {addr}")

            num_steps = self.world_size
            while True:
                self.listen_mtx.acquire()
                with self.cond:
                    if self.stop_flag:
                        break
                        
                for step in range(2 * num_steps - 2):
                    local_chunk_index = (self.rank - step - 1) % self.world_size
                    template_chunk = self.chunks[local_chunk_index]

                    raw_bytes = _recv_bytes(self.listener_sock)
                    chunk = _unpack_bytes(raw_bytes, template_chunk)

                    with self.cond:
                        self.listener_cnt += 1
                        if step >= num_steps - 1:
                            # All-gather phase
                            self.chunks[(self.rank - step - 1) % self.world_size] = chunk
                        else:
                            # Scatter-reduce phase
                            local_chunk_index = (self.rank - step - 1) % self.world_size
                            for i in range(len(self.chunks[local_chunk_index])):
                                self.chunks[local_chunk_index][i] += chunk[i]
                        self.buffer_full = True
                        self.cond.notify_all()

                with self.cond:
                    self.iteration_cnt += 1
                    self.cond.notify_all()

    def _sender_thread(self):
        self.sender_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sender_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        right_ip = self.right_neighbor_info["ip"]
        right_port = self.connect_port
        self._log(f"Connecting to right neighbor at {right_ip}:{right_port}...")

        while True:
            try:
                self.sender_sock.connect((right_ip, right_port))
                break
            except ConnectionRefusedError:
                time.sleep(1)

        self._log(f"Connected to right neighbor")
        
        while True:
            self.snd_mtx.acquire()
            with self.cond:
                if self.stop_flag:
                    break

            num_steps = self.world_size
            for step in range(2 * num_steps - 2):
                with self.cond:
                    chunk_to_send = self.chunks[(self.rank - step) % self.world_size]
                    _send_chunk(self.sender_sock, chunk_to_send)
                    while not self.buffer_full and step != (2 * num_steps - 2) - 1:
                        self.cond.wait()
                    if (self.listener_cnt - 1) > step:
                        continue
                    self.buffer_full = False

            with self.cond:
                self.iteration_cnt += 1
                self.cond.notify_all()

    def allreduce(self, data_list):
        """
        Average gradients across all nodes using ring allreduce.
        
        Args:
            data_list: List of numpy arrays (gradients) to average
            
        Returns:
            List of averaged numpy arrays in the same order
        """
        # Split into chunks
        self.chunks = _chunk_list(data_list, self.world_size)
        self.listen_mtx.release()
        self.snd_mtx.release()

        # Wait for both threads to complete
        with self.cond:
            while self.iteration_cnt != 2:
                self.cond.wait()
            self.iteration_cnt = 0
            self.listener_cnt = 0
            self.buffer_full = False

        # Average each chunk
        averaged_chunks = [
            [arr / self.world_size for arr in chunk]
            for chunk in self.chunks
        ]

        # Reconstruct full list
        averaged_gradients = []
        for i in range(self.world_size):
            averaged_gradients.extend(averaged_chunks[i])

        if len(averaged_gradients) != len(data_list):
            raise ValueError(f"Size mismatch: got {len(averaged_gradients)}, expected {len(data_list)}")

        return averaged_gradients

    def close(self):
        """Clean up sockets and stop threads"""
        if self.sender_sock:
            self.sender_sock.close()
        if self.listener_sock:
            self.listener_sock.close()
        with self.cond:
            self.stop_flag = True
            try:
                self.snd_mtx.release()
            except RuntimeError:
                pass
            try:
                self.listen_mtx.release()
            except RuntimeError:
                pass
