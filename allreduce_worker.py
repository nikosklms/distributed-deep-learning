# allreduce_worker.py
import socket
import threading
import time
import numpy as np
import sys
import pickle
import hashlib

from load_data import load_mnist
from core import Linear, ReLU, CrossEntropyLoss, SGD

# --- Training Config ---
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.1 # The 0.1 you found works well!

DISCOVERY_IP = "0.0.0.0"  # IP of discovery server
DISCOVERY_PORT = 5000

# Choose port
BASE_PORT = 6000

cnt_top = 0
cnt_pot = 0

def send_chunk(sock, chunk_list):
    """Sends a list of numpy arrays as a single, raw byte buffer."""
    all_bytes = b''.join([arr.astype(np.float32).tobytes() for arr in chunk_list])
    
    sock.sendall(len(all_bytes).to_bytes(8, 'big'))
    
    sock.sendall(all_bytes)

def recv_bytes(sock):
    """Receives a byte buffer sent by send_chunk."""
    header = b''
    while len(header) < 8:
        chunk = sock.recv(8 - len(header))
        if not chunk:
            return None
        header += chunk

    data_length = int.from_bytes(header, 'big')
    if data_length <= 0 or data_length > 1 << 30: # 1GB limit
        raise ValueError(f"Invalid data length: {data_length}")

    # We pre-allocate the entire buffer
    data_bytes = bytearray(data_length)
    view = memoryview(data_bytes)
    
    while data_length > 0:
        # sock.recv_into() writes directly into our buffer (fast!)
        nbytes = sock.recv_into(view, data_length)
        if not nbytes:
            raise EOFError("Connection broken while receiving data")
        
        view = view[nbytes:] # Move the view "cursor" forward
        data_length -= nbytes

    # Return the completed bytearray
    return data_bytes

def unpack_bytes(data_bytes, template_chunk_list):
    """Unpacks raw bytes into a new list of arrays based on a template."""
    received_chunk = []
    offset = 0
    
    for template_arr in template_chunk_list:
        num_bytes = np.prod(template_arr.shape) * 4 # 4 bytes for float32
        
        arr_bytes = data_bytes[offset : offset + num_bytes]
        
        new_arr = np.frombuffer(arr_bytes, dtype=np.float32).reshape(template_arr.shape).copy()
        received_chunk.append(new_arr)
        
        offset += num_bytes
    
    # Sanity check
    if offset != len(data_bytes):
        raise ValueError(f"Unpack size mismatch. Expected {offset} bytes, got {len(data_bytes)}")

    return received_chunk

def chunk_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

# --- 4. The AllReduce Function ---
# This class wraps all our ring logic
class RingAllReducer:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

        # Get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        finally:
            s.close()

        self.listen_port = BASE_PORT + self.rank
        print(f"My listen port is {self.listen_port}")

        # Discovery info
        self.my_info = {"ip": local_ip, "port": self.listen_port, "rank": self.rank}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((DISCOVERY_IP, DISCOVERY_PORT))
            s.sendall(pickle.dumps(self.my_info))
            data = s.recv(4096)
            all_clients = pickle.loads(data)

        print(f"[Node {self.rank}] Discovered clients:", all_clients)

        # Sort by rank
        all_clients.sort(key=lambda x: x["rank"])
        my_index = next(i for i, c in enumerate(all_clients) if c["rank"] == self.rank)

        self.right_neighbor_info = all_clients[(my_index + 1) % world_size]
        self.left_neighbor_info = all_clients[(my_index - 1 + world_size) % world_size]
        self.connect_port = self.right_neighbor_info["port"]

        self.listener_sock = None
        self.sender_sock = None

        print(f"[Node {self.rank}] Left neighbor: {self.left_neighbor_info}")
        print(f"[Node {self.rank}] Right neighbor: {self.right_neighbor_info}")

        self.chunks = []
        self.buffer_full = False
        self.cond = threading.Condition()
        self.snd_mtx = threading.Lock()
        self.snd_mtx.acquire()
        self.listen_mtx = threading.Lock()
        self.listen_mtx.acquire()
        self.itteration_cnt = 0
        self.stop_flag = False

        self.listener_cnt = 0

        # Start threads
        self.listener = threading.Thread(target=self._listen_thread, daemon=True)
        self.listener.start()
        self.sender = threading.Thread(target=self._sender_thread, daemon=True)
        self.sender.start()

    def _listen_thread(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.my_info["ip"], self.my_info["port"]))
            s.listen()
            print(f"[Node {self.rank}] Listening on {self.my_info['ip']}:{self.my_info['port']}...")
            conn, addr = s.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.listener_sock = conn
            print(f"[Node {self.rank}] Got connection from left neighbor {addr}")

            num_steps = self.world_size
            counter = 0
            while True:
                counter += 1
                print(f"listener: iteration {counter}")
                self.listen_mtx.acquire()
                with self.cond:
                    if self.stop_flag:
                        break
                for step in range(2 * num_steps - 2):
                    local_chunk_index = (self.rank - step - 1) % self.world_size
                    template_chunk = self.chunks[local_chunk_index]

                    # 2. Receive raw bytes
                    raw_bytes = recv_bytes(self.listener_sock)
                    chunk = unpack_bytes(raw_bytes, template_chunk)
                    print(f"{cnt_pot} {cnt_top} [Node {self.rank}] Received chunk at step {step}")

                    with self.cond:
                        self.listener_cnt += 1
                        if step >= num_steps - 1:
                            self.chunks[(self.rank - step - 1) % self.world_size] = chunk
                        else:
                            # Scatter-Reduce phase (sum the arrays in the chunk)                            
                            local_chunk_index = (self.rank - step - 1) % self.world_size
                            
                            # Loop through each array in the chunk list and add in-place
                            for i in range(len(self.chunks[local_chunk_index])):
                                self.chunks[local_chunk_index][i] += chunk[i]
                        self.buffer_full = True
                        self.cond.notify_all()

                with self.cond:
                    self.itteration_cnt += 1
                    self.cond.notify_all()

            print("Listener ended")

    def _sender_thread(self):
        self.sender_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sender_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        right_ip = self.right_neighbor_info["ip"]
        right_port = self.connect_port
        print(f"[Node {self.rank}] Connecting to right neighbor at {right_ip}:{right_port}...")

        while True:
            try:
                self.sender_sock.connect((right_ip, right_port))
                break
            except ConnectionRefusedError:
                time.sleep(1)

        print(f"[Node {self.rank}] Connected to right neighbor at {right_ip}:{right_port}")
        counter = 0
        while True:
            counter += 1
            print(f"sender: iteration {counter}")
            self.snd_mtx.acquire()
            with self.cond:
                if self.stop_flag:
                    break

            num_steps = self.world_size
            for step in range(2 * num_steps - 2):
                with self.cond:
                    print(f"{cnt_pot} {cnt_top} [Node {self.rank}] Sending chunk at step {step}")
                    chunk_to_send = self.chunks[(self.rank - step) % self.world_size]
                    send_chunk(self.sender_sock, chunk_to_send)
                    while not self.buffer_full and step != (2 * num_steps - 2) - 1:
                        print("sender here!!!")
                        self.cond.wait()
                    if (self.listener_cnt-1) > step:
                        continue
                    self.buffer_full = False

            with self.cond:
                self.itteration_cnt += 1
                self.cond.notify_all()

        print("Sender ended")

    def allreduce(self, data_list):
        # Step 1: split into chunks
        self.chunks = chunk_list(data_list, self.world_size)
        self.listen_mtx.release()
        self.snd_mtx.release()

        with self.cond:
            while self.itteration_cnt != 2:
                self.cond.wait()
            self.itteration_cnt = 0
            self.listener_cnt = 0
            self.buffer_full = False

        print("Both listener and sender ended")
        print(f"Final chunk list lengths: {[len(c) for c in self.chunks]}")

        # Step 2: average each chunk
        averaged_chunks = [
            [arr / self.world_size for arr in chunk]
            for chunk in self.chunks
        ]

        # Step 3: reconstruct full list using same logic as chunk_list()
        k, m = divmod(len(data_list), self.world_size)
        averaged_gradients = []
        for i in range(self.world_size):
            averaged_gradients.extend(averaged_chunks[i])

        # Debug info
        print(f"Reconstructed averaged gradients len={len(averaged_gradients)}, expected={len(data_list)}")

        # Sanity check
        if len(averaged_gradients) != len(data_list):
            raise ValueError(f"Size mismatch: got {len(averaged_gradients)}, expected {len(data_list)}")

        return averaged_gradients

    def close(self):
        self.sender_sock.close()
        self.listener_sock.close()
        with self.cond:
            self.stop_flag = True
            self.snd_mtx.release()
            self.listen_mtx.release()

        self.listener.join()
        self.sender.join()

def hash_list_of_arrays(arr_list):
    """Creates a SHA256 hash of a list of numpy arrays."""
    m = hashlib.sha256()
    for arr in arr_list:
        # Use .tobytes() to get the raw data in a consistent way
        m.update(arr.astype(np.float32).tobytes())
    return m.hexdigest()

def main(rank, world_size):
    # --- Load and SHARD the data ---
    print(f"[Node {rank}] Loading and sharding data...")
    X_train_all, y_train_all, X_test, y_test = load_mnist()
    
    # Split the 60,000 samples among all workers
    num_samples = X_train_all.shape[0]
    samples_per_node = num_samples // world_size
    
    start_idx = rank * samples_per_node
    end_idx = start_idx + samples_per_node
    
    X_train = X_train_all[start_idx:end_idx]
    y_train = y_train_all[start_idx:end_idx]
    
    num_train_samples = X_train.shape[0]
    num_batches = num_train_samples // BATCH_SIZE
    print(f"[Node {rank}] Training on {num_train_samples} samples.")

    # --- Build the LOCAL Model ---
    layer1 = Linear(input_size=784, output_size=128, seed=42)
    activation1 = ReLU()
    layer2 = Linear(input_size=128, output_size=10, seed=42)
    loss_function = CrossEntropyLoss()
    
    # Each node has its own optimizer
    parameters = [layer1, layer2]
    optimizer = SGD(parameters=parameters, learning_rate=LEARNING_RATE)

    # --- Connect the Ring ---
    comm = RingAllReducer(rank, world_size)

    start = time.time()
    avg1 = 0
    avg2 = 0
    # --- The Training Loop ---
    for epoch in range(EPOCHS):
        global cnt_pot
        cnt_pot = epoch
        # We don't need to shuffle, since data is already pre-sharded
        running_loss = 0.0
        
        for i in range(num_batches):
            global cnt_top
            cnt_top = i
            X_batch = X_train[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            y_batch = y_train[i*BATCH_SIZE : (i+1)*BATCH_SIZE]

            start1 = time.time()
            # --- 1. FORWARD PASS ---
            out1 = layer1.forward(X_batch)
            out_relu = activation1.forward(out1)
            logits = layer2.forward(out_relu)
            loss = loss_function.forward(logits, y_batch)
            running_loss += loss
            
            # --- 2. BACKWARD PASS ---
            optimizer.zero_grad()
            d_logits = loss_function.backward()
            d_layer2 = layer2.backward(d_logits)
            d_relu = activation1.backward(d_layer2)
            d_layer1 = layer1.backward(d_relu)
            
            end1 = time.time()
            avg1 += end1 - start1
            # --- 3. ALLREDUCE GRADIENTS ---
            # Create a list of all gradients
            gradients = [layer1.d_weights, layer1.d_biases, layer2.d_weights, layer2.d_biases]
            
            start2 = time.time()
            # Call the function to average them with our neighbor
            avg_gradients = comm.allreduce(gradients)
            end2 = time.time()
            avg2 += end2 - start2

            avg_grad_hash = hash_list_of_arrays(avg_gradients)
            print(f"[Node {rank}] Iter {i}, Avg Grad Hash: {avg_grad_hash}")
            
            # Put the new averaged gradients back into our layers
            layer1.d_weights, layer1.d_biases, layer2.d_weights, layer2.d_biases = avg_gradients
            
            # --- 4. UPDATE WEIGHTS ---
            optimizer.step() # We are all stepping with the SAME gradient
        
        avg_loss = running_loss / num_batches
        print(f"[Node {rank}] Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")

        h = hashlib.md5(np.concatenate([p.weights.flatten() for p in [layer1, layer2]]).tobytes()).hexdigest()
        print(f"[Node {rank}] Param hash:", h)
    
    print(f"AVG1 is {avg1/(EPOCHS*num_batches)} and AVG2 is {avg2/(EPOCHS*num_batches)}")

    comm.close()

    # --- Test the Network (Only Node 0 does this) ---
    if rank == 0:
        print("Training finished. Evaluating on test set...")
        out1 = layer1.forward(X_test)
        out_relu = activation1.forward(out1)
        logits = layer2.forward(out_relu)
        predictions = np.argmax(logits, axis=1)
        accuracy = np.mean(predictions == y_test)
        print(f"\n[Node 0] FINAL Test Accuracy: {accuracy * 100:.2f}%")

        end = time.time()
        print(f"Elapsed time: {end - start:.6f} seconds")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 allreduce_worker.py <rank> <world_size>")
        sys.exit(1)
        
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    
    main(rank, world_size)