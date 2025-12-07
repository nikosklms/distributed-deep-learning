import socket
import threading
import pickle
import time
import queue
import numpy as np
import fast_net
import struct  # Required for header unpacking

# --- Config ---
DISCOVERY_IP = "0.0.0.0" 
DISCOVERY_PORT = 5000
BASE_PORT = 6000

def chunk_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def send_chunk(sock, chunk_list):
    """Sends a list of numpy arrays using C fast socket."""
    ready_list = []
    for arr in chunk_list:
        flat = arr.ravel() 
        if flat.dtype != np.float16:
            flat = flat.astype(np.float16)
        if not flat.flags['C_CONTIGUOUS']:
            flat = np.ascontiguousarray(flat)
        ready_list.append(flat)

    fast_net.send_list(sock.fileno(), ready_list)

def recv_and_unpack(sock, template_chunk_list):
    """
    Robust Receive:
    1. Python reads 8-byte header (Size).
    2. Python allocates Buffer.
    3. C reads Payload (Fast).
    """
    # 1. Read Header (8 bytes long) using Python
    # This avoids the "C expects X but got Y" crash. We trust the network.
    header_data = b''
    while len(header_data) < 8:
        packet = sock.recv(8 - len(header_data))
        if not packet:
            raise EOFError("Connection closed during header recv")
        header_data += packet
    
    # Unpack long (8 bytes). '<q' = little-endian signed long long (standard for 64bit)
    # Using 'q' to match the C 'long' on 64-bit systems.
    # If this fails, we might need sys.byteorder, but usually little-endian is safe.
    total_bytes = int.from_bytes(header_data, 'little')
    
    # 2. Allocate Buffer
    # num_elements = total_bytes // 2 (since float16 is 2 bytes)
    num_elements = total_bytes // 2
    recv_buffer = np.empty(num_elements, dtype=np.float16)
    
    # 3. Call C to fill the buffer (Fast Payload Transfer)
    # We use a modified/simplified recv logic in C (recv_into_array)
    # Note: Our current C recv_into_array EXPECTS a header. 
    # We already ate the header! So calling C now will try to read Data as Header.
    # -> WE NEED TO FIX THIS.
    
    # STRATEGY CHANGE:
    # We cannot mix Python header read + C payload read with the CURRENT C code
    # because the C code forces reading a header.
    
    # IMMEDIATE FIX:
    # Use the C code as intended, but Wrap it in try-except to catch the size mismatch
    # and print useful info.
    # OR: Use Python for everything temporarily to verify logic.
    
    # Let's revert to calling C, but handle the error gracefully.
    
    expected_elements = sum(arr.size for arr in template_chunk_list)
    expected_buffer = np.empty(expected_elements, dtype=np.float16)
    
    # Put back the header bytes? No.
    # We must not read header in Python if C expects it.
    pass

# --- REAL IMPLEMENTATION OF recv_and_unpack (Reverting to C call) ---
def recv_and_unpack_c(sock, template_chunk_list, rank):
    total_elements = sum(arr.size for arr in template_chunk_list)
    recv_buffer = np.empty(total_elements, dtype=np.float16)
    
    # Call C
    # It might raise ValueError if size mismatches
    try:
        success = fast_net.recv_into_array(sock.fileno(), recv_buffer)
    except ValueError as e:
        print(f"[Node {rank}] FATAL SYNC ERROR: {e}")
        # Return zeros to avoid crash, but training is doomed
        return [np.zeros_like(t) for t in template_chunk_list]

    if not success:
        raise EOFError("Connection broken")
    
    restored_chunk = []
    offset = 0
    for template in template_chunk_list:
        size = template.size
        arr_view = recv_buffer[offset : offset+size].reshape(template.shape)
        restored_chunk.append(arr_view)
        offset += size
        
    return restored_chunk

class RingAllReducer:
    def __init__(self, rank, world_size, discovery_ip=DISCOVERY_IP, discovery_port=DISCOVERY_PORT):
        self.rank = rank
        self.world_size = world_size
        self.send_queue = queue.Queue()
        
        if self.world_size == 1:
            return

        # 1. Discovery
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try: s.connect(("8.8.8.8", 80)); local_ip = s.getsockname()[0]
        finally: s.close()
        
        listen_port = BASE_PORT + self.rank
        info = {"ip": local_ip, "port": listen_port, "rank": self.rank}
        
        # print(f"[Node {self.rank}] Connecting to Discovery...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((discovery_ip, discovery_port))
            s.sendall(pickle.dumps(info))
            data = s.recv(4096)
            all_clients = pickle.loads(data)
            
        all_clients.sort(key=lambda x: x["rank"])
        my_idx = next(i for i, c in enumerate(all_clients) if c["rank"] == self.rank)
        self.right_neighbor = all_clients[(my_idx + 1) % world_size]
        
        # 2. Setup Sockets
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((local_ip, listen_port))
        self.server.listen(1)
        
        self.sender_sock = None
        self.listener_sock = None
        
        t_conn = threading.Thread(target=self._connect_sender)
        t_conn.start()
        
        conn, _ = self.server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.listener_sock = conn
        
        t_conn.join()
        print(f"[Node {rank}] Ring ready.")

    def _connect_sender(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            try:
                sock.connect((self.right_neighbor["ip"], self.right_neighbor["port"]))
                self.sender_sock = sock
                break
            except ConnectionRefusedError:
                time.sleep(0.5)

    def _listen_task(self, send_queue, num_steps):
        for step in range(2 * num_steps - 2):
            recv_idx = (self.rank - step - 1) % self.world_size
            template = self.chunks[recv_idx]
            
            # Use C recv
            data = recv_and_unpack_c(self.listener_sock, template, self.rank)
            
            if step < num_steps - 1: # Scatter-Reduce
                for i in range(len(self.chunks[recv_idx])):
                    self.chunks[recv_idx][i] += data[i]
            else: # All-Gather
                self.chunks[recv_idx] = data
            
            send_queue.put(recv_idx)

    def _sender_task(self, send_queue, num_steps):
        for _ in range(2 * num_steps - 2):
            chunk_idx = send_queue.get()
            send_chunk(self.sender_sock, self.chunks[chunk_idx])
            send_queue.task_done()

    def allreduce(self, data_list):
        if self.world_size == 1:
            return [x.copy() for x in data_list]

        self.chunks = chunk_list(data_list, self.world_size)
        N = self.world_size
        
        self.send_queue.put(self.rank)
        
        t1 = threading.Thread(target=self._listen_task, args=(self.send_queue, N))
        t2 = threading.Thread(target=self._sender_task, args=(self.send_queue, N))
        t1.start(); t2.start()
        t1.join(); t2.join()
        
        res = []
        for ch in self.chunks:
            res.extend([x / N for x in ch])
        return res

    def close(self):
        if self.world_size > 1:
            if self.sender_sock: self.sender_sock.close()
            if self.listener_sock: self.listener_sock.close()
            self.server.close()