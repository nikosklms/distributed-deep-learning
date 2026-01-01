# allreduce_worker.py
import socket
import threading
import time
import numpy as np
import sys
import pickle
import hashlib
import cupy as cp
import fast_net
import os
import queue

from load_data import load_mnist
from core_gpu import LinearGPU, ReLUGPU, CrossEntropyLossGPU, SGD_GPU

# --- Training Config ---
EPOCHS = 10 * int(sys.argv[2])
BATCH_SIZE = 0
LEARNING_RATE = 0

DISCOVERY_IP = os.getenv("DISCOVERY_IP", "0.0.0.0")  # IP of discovery server
DISCOVERY_PORT = 5000

# Choose port
BASE_PORT = 6000

# def send_chunk(sock, chunk_list):
#     """Sends a list of numpy arrays as a single, raw byte buffer."""
#     all_bytes = b''.join([arr.astype(np.float16).tobytes() for arr in chunk_list])
    
#     sock.sendall(len(all_bytes).to_bytes(8, 'big'))
    
#     sock.sendall(all_bytes)

def send_chunk(sock, chunk_list):
    """Sends a list of numpy arrays using C fast socket (No Concatenation)."""
    
    # Προετοιμασία: Βεβαιωνόμαστε ότι κάθε κομμάτι είναι float16 και contiguous.
    # Αυτό είναι πολύ φτηνό σε σχέση με το να ενώσουμε τα πάντα.
    ready_list = []
    for arr in chunk_list:
        # Flatten (αν δεν είναι ήδη) -> Επιστρέφει συνήθως contiguous copy ή view
        flat = arr.ravel() 
        
        # Μετατροπή σε float16 αν χρειάζεται
        if flat.dtype != np.float16:
            flat = flat.astype(np.float16)
            
        # Βεβαιωνόμαστε ότι είναι συνεχόμενο στη μνήμη (για να το διαβάσει η C)
        if not flat.flags['C_CONTIGUOUS']:
            flat = np.ascontiguousarray(flat)
            
        ready_list.append(flat)

    # 2. ΚΑΛΟΥΜΕ ΤΗ C ΣΥΝΑΡΤΗΣΗ ΠΟΥ ΔΕΧΕΤΑΙ ΛΙΣΤΑ
    fast_net.send_list(sock.fileno(), ready_list)

# def recv_bytes(sock):
#     """Receives a byte buffer sent by send_chunk."""
#     header = b''
#     while len(header) < 8:
#         chunk = sock.recv(8 - len(header))
#         if not chunk:
#             return None
#         header += chunk

#     data_length = int.from_bytes(header, 'big')
#     if data_length <= 0 or data_length > 1 << 30: # 1GB limit
#         raise ValueError(f"Invalid data length: {data_length}")

#     # We pre-allocate the entire buffer
#     data_bytes = bytearray(data_length)
#     view = memoryview(data_bytes)
    
#     while data_length > 0:
#         # sock.recv_into() writes directly into our buffer (fast!)
#         nbytes = sock.recv_into(view, data_length)
#         if not nbytes:
#             raise EOFError("Connection broken while receiving data")
        
#         view = view[nbytes:] # Move the view "cursor" forward
#         data_length -= nbytes

#     # Return the completed bytearray
#     return data_bytes

# def unpack_bytes(data_bytes, template_chunk_list):
#     """Unpacks raw bytes into a new list of arrays based on a template."""
#     received_chunk = []
#     offset = 0
    
#     for template_arr in template_chunk_list:
#         num_bytes = np.prod(template_arr.shape) * 2 # 4 bytes for float32
        
#         arr_bytes = data_bytes[offset : offset + num_bytes]
        
#         new_arr = np.frombuffer(arr_bytes, dtype=np.float16).reshape(template_arr.shape).copy()
#         received_chunk.append(new_arr)
        
#         offset += num_bytes
    
#     # Sanity check
#     if offset != len(data_bytes):
#         raise ValueError(f"Unpack size mismatch. Expected {offset} bytes, got {len(data_bytes)}")

#     return received_chunk

def recv_and_unpack(sock, template_chunk_list):
    """Receives data directly into a numpy array using C fast socket."""
    
    # 1. Υπολογίζουμε πόσα στοιχεία (float16 numbers) περιμένουμε συνολικά
    total_elements = sum(arr.size for arr in template_chunk_list)
    
    # 2. Φτιάχνουμε έναν κενό "μεγάλο" πίνακα για να υποδεχτεί τα δεδομένα
    # Χρησιμοποιούμε np.empty για ταχύτητα (δεν μηδενίζει τη μνήμη)
    recv_buffer = np.empty(total_elements, dtype=np.float16)
    
    # 3. ΚΑΛΟΥΜΕ ΤΗ C ΣΥΝΑΡΤΗΣΗ ΝΑ ΤΟ ΓΕΜΙΣΕΙ
    # Αυτή θα μπλοκάρει μέχρι να έρθουν όλα τα bytes
    # Επιστρέφει True αν πέτυχε, False αν έκλεισε η σύνδεση
    success = fast_net.recv_into_array(sock.fileno(), recv_buffer)
    
    if not success:
        raise EOFError("Connection broken while receiving data in C module")
    
    # 4. Κόβουμε τον μεγάλο πίνακα στα μικρά κομμάτια (Reshape / View)
    # Εδώ δεν γίνεται αντιγραφή μνήμης, απλά φτιάχνουμε "views"
    restored_chunk = []
    offset = 0
    for template in template_chunk_list:
        size = template.size
        # Παίρνουμε το κομμάτι και του δίνουμε το σωστό σχήμα (π.χ. 784x128)
        arr_view = recv_buffer[offset : offset+size].reshape(template.shape)
        restored_chunk.append(arr_view)
        offset += size
        
    return restored_chunk


def chunk_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

# --- 4. The AllReduce Function ---
# This class wraps all our ring logic 

class RingAllReducer:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

        # --- 1. Discovery & Setup ---
        # Βρίσκουμε την τοπική IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        finally:
            s.close()

        self.listen_port = BASE_PORT + self.rank
        self.my_info = {"ip": local_ip, "port": self.listen_port, "rank": self.rank}
        
        print(f"[Node {self.rank}] Setup: Listening on {local_ip}:{self.listen_port}")

        # Συνδεόμαστε στον Discovery Server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((DISCOVERY_IP, DISCOVERY_PORT))
            s.sendall(pickle.dumps(self.my_info))
            data = s.recv(4096)
            all_clients = pickle.loads(data)

        # Ταξινόμηση και εύρεση γειτόνων
        all_clients.sort(key=lambda x: x["rank"])
        my_index = next(i for i, c in enumerate(all_clients) if c["rank"] == self.rank)
        
        self.right_neighbor = all_clients[(my_index + 1) % world_size]
        # (Ο αριστερός γείτονας δεν χρειάζεται ως info, απλά δεχόμαστε σύνδεση από αυτόν)

        # --- 2. Permanent Connection Establishment ---
        # Ανοίγουμε τα sockets ΜΙΑ φορά στην αρχή για να γλιτώσουμε χρόνο αργότερα.
        
        # A. Listener Socket (Server)
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((local_ip, self.listen_port))
        self.server_sock.listen(1)
        
        # B. Connection Logic (Async για αποφυγή Deadlock)
        # Ξεκινάμε thread να συνδεθεί στον δεξιά, ενώ εμείς περιμένουμε τον αριστερά.
        self.sender_sock = None
        self.listener_sock = None
        
        connect_thread = threading.Thread(target=self._connect_to_right)
        connect_thread.start()
        
        # Περιμένουμε τον αριστερό να συνδεθεί σε εμάς
        print(f"[Node {self.rank}] Waiting for left neighbor...")
        conn, addr = self.server_sock.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) # Σημαντικό για speed
        self.listener_sock = conn
        print(f"[Node {self.rank}] Accepted connection from {addr}")
        
        # Περιμένουμε να ολοκληρωθεί και η δική μας σύνδεση στον δεξιά
        connect_thread.join()
        print(f"[Node {self.rank}] Ring established!")

        # State για το AllReduce
        self.chunks = []

    def _connect_to_right(self):
        """Προσπαθεί επίμονα να συνδεθεί στον δεξιά γείτονα."""
        target_ip = self.right_neighbor["ip"]
        target_port = self.right_neighbor["port"]
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        while True:
            try:
                sock.connect((target_ip, target_port))
                self.sender_sock = sock
                print(f"[Node {self.rank}] Connected to right neighbor {target_ip}:{target_port}")
                break
            except ConnectionRefusedError:
                # Ο γείτονας δεν έχει σηκώσει ακόμα το server socket, περιμένουμε λίγο
                time.sleep(0.1)

    def _listen_task(self, send_queue, num_steps):
        """
        Thread που ακούει δεδομένα.
        Μόλις λάβει και επεξεργαστεί ένα κομμάτι, ενημερώνει την ουρά 
        ότι 'το τάδε κομμάτι είναι έτοιμο να σταλεί στον επόμενο'.
        """
        for step in range(2 * num_steps - 2):
            # Υπολογισμός: Ποιο κομμάτι (chunk index) περιμένουμε σε αυτό το βήμα;
            # Στο Ring-AllReduce τα chunks έρχονται ανάποδα από τη φορά του ρολογιού.
            recv_chunk_idx = (self.rank - step - 1) % self.world_size
            
            # 1. Λήψη (Blocking C call - High Performance)
            # Χρειαζόμαστε το template για να ξέρει η C πόσα bytes να περιμένει
            template = self.chunks[recv_chunk_idx]
            received_data = recv_and_unpack(self.listener_sock, template)
            
            # 2. Επεξεργασία (Compute)
            if step < num_steps - 1:
                # Φάση Scatter-Reduce: Προσθέτουμε τα gradients
                for i in range(len(self.chunks[recv_chunk_idx])):
                    self.chunks[recv_chunk_idx][i] += received_data[i]
            else:
                # Φάση All-Gather: Αντικαθιστούμε (λαμβάνουμε το τελικό άθροισμα)
                self.chunks[recv_chunk_idx] = received_data
            
            # 3. Ειδοποίηση Sender
            # "Το κομμάτι recv_chunk_idx ενημερώθηκε, στείλ' το παρακάτω!"
            send_queue.put(recv_chunk_idx)

    def _sender_task(self, send_queue, num_steps):
        """
        Thread που στέλνει δεδομένα.
        Περιμένει εντολές από την ουρά.
        """
        for _ in range(2 * num_steps - 2):
            # 1. Περιμένουμε να γίνει διαθέσιμο ένα κομμάτι (Blocking)
            chunk_idx_to_send = send_queue.get()
            
            # 2. Αποστολή (Blocking C call - High Performance)
            send_chunk(self.sender_sock, self.chunks[chunk_idx_to_send])
            
            # Σηματοδοτούμε ότι τελειώσαμε
            send_queue.task_done()

    def allreduce(self, data_list):
        # 1. Προετοιμασία
        self.chunks = chunk_list(data_list, self.world_size)
        num_steps = self.world_size
        
        # Η ουρά επικοινωνίας μεταξύ Listener -> Sender για ΑΥΤΟΝ τον γύρο
        send_queue = queue.Queue()
        
        # 2. Kickstart (Έναρξη)
        # Στο βήμα 0, πρέπει να στείλουμε το δικό μας κομμάτι.
        # Το βάζουμε στην ουρά για να το πάρει ο Sender και να ξεκινήσει το ντόμινο.
        my_chunk_idx = self.rank
        send_queue.put(my_chunk_idx)
        
        # 3. Εκκίνηση Threads
        t_listen = threading.Thread(target=self._listen_task, args=(send_queue, num_steps))
        t_send = threading.Thread(target=self._sender_task, args=(send_queue, num_steps))
        
        t_listen.start()
        t_send.start()
        
        # 4. Αναμονή Τερματισμού
        t_listen.join()
        t_send.join()
        
        # 5. Ανασυγκρότηση και Μέσος Όρος
        averaged_gradients = []
        
        # Ενώνουμε τα chunks και διαιρούμε με τον αριθμό των workers (Average)
        for chunk in self.chunks:
            for arr in chunk:
                arr /= self.world_size # In-place division (γρήγορο)
                averaged_gradients.append(arr)
            
        return averaged_gradients

    def close(self):
        if self.sender_sock: self.sender_sock.close()
        if self.listener_sock: self.listener_sock.close()
        self.server_sock.close()


def hash_list_of_arrays(arr_list):
    """Creates a SHA256 hash of a list of numpy arrays."""
    m = hashlib.sha256()
    for arr in arr_list:
        # Use .tobytes() to get the raw data in a consistent way
        m.update(arr.astype(np.float32).tobytes())
    return m.hexdigest()

def main(rank, world_size, batch_size, hidden_size, learning_rate):
    # --- Load and SHARD the data ---
    print(f"[Node {rank}] Loading and sharding data...")
    X_train_all, y_train_all, X_test, y_test = load_mnist()

    # mix-blend the data so all workers have the same sample,
    # but the same "random" state
    indices = np.arange(X_train_all.shape[0])
    np.random.seed(42)
    np.random.shuffle(indices)
    
    X_train_all = X_train_all[indices]
    y_train_all = y_train_all[indices]

    # Split the 60,000 samples among all workers
    num_samples = X_train_all.shape[0]
    samples_per_node = num_samples // world_size
    
    start_idx = rank * samples_per_node
    end_idx = start_idx + samples_per_node
    
    X_train = X_train_all[start_idx:end_idx]
    y_train = y_train_all[start_idx:end_idx]

    print(f"[Node {rank}] Moving data to GPU VRAM...")
    X_train_gpu = cp.asarray(X_train)
    y_train_gpu = cp.asarray(y_train)

    num_train_samples = X_train.shape[0]
    
    num_batches = num_train_samples // batch_size
    
    print(f"[Node {rank}] Training on {num_train_samples} samples.")
    print(f"[Node {rank}] Config: Batch={batch_size}, Hidden={hidden_size}, LR={learning_rate}")

    # --- Build the LOCAL Model ---
    layer1 = LinearGPU(input_size=784, output_size=hidden_size, seed=42)
    activation1 = ReLUGPU()
    layer2 = LinearGPU(input_size=hidden_size, output_size=10, seed=42)
    loss_function = CrossEntropyLossGPU()
    
    # Each node has its own optimizer
    parameters = [layer1, layer2]
    optimizer = SGD_GPU(parameters=parameters, learning_rate=learning_rate)

    # --- Connect the Ring ---
    comm = RingAllReducer(rank, world_size)

    start = time.time()
    avg1 = 0
    avg2 = 0
    # --- The Training Loop ---
    for epoch in range(EPOCHS):
        running_loss = 0.0
        
        for i in range(num_batches):
            
            start_batch = i * batch_size
            end_batch = (i + 1) * batch_size
            
            X_batch = X_train_gpu[start_batch : end_batch]
            y_batch = y_train_gpu[start_batch : end_batch]

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
            
            g1_w = layer1.d_weights.get()
            g1_b = layer1.d_biases.get()
            g2_w = layer2.d_weights.get()
            g2_b = layer2.d_biases.get()

            end1 = time.time()
            avg1 += end1 - start1

            # --- 3. ALLREDUCE GRADIENTS ---
            gradients_cpu_list = [g1_w, g1_b, g2_w, g2_b]
            
            start2 = time.time()
            avg_gradients_cpu = comm.allreduce(gradients_cpu_list)
            end2 = time.time()
            avg2 += end2 - start2

            layer1.d_weights = cp.asarray(avg_gradients_cpu[0])
            layer1.d_biases = cp.asarray(avg_gradients_cpu[1])
            layer2.d_weights = cp.asarray(avg_gradients_cpu[2])
            layer2.d_biases = cp.asarray(avg_gradients_cpu[3])
            
            # --- 4. UPDATE WEIGHTS ---
            optimizer.step()
        
        avg_loss = running_loss / num_batches
        print(f"[Node {rank}] Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f} {running_loss} {num_batches}")

    print(f"AVG1 (Compute) is {avg1/(EPOCHS*num_batches):.5f} and AVG2 (Comm) is {avg2/(EPOCHS*num_batches):.5f}")

    comm.close()

    # --- Test the Network (Only Node 0 does this) ---
    if rank == 0:
        print("Evaluating on test set...")
        X_test_gpu = cp.asarray(X_test)
        y_test_gpu = cp.asarray(y_test)

        out1 = layer1.forward(X_test_gpu)
        out_relu = activation1.forward(out1)
        logits = layer2.forward(out_relu)
        
        predictions = cp.argmax(logits, axis=1)
        accuracy = cp.mean(predictions == y_test_gpu)
        
        print(f"\n[Node 0] FINAL Test Accuracy: {float(accuracy) * 100:.2f}%")
        end = time.time()
        print(f"Elapsed time: {end - start:.6f} seconds")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 allreduce_worker.py <rank> <world_size> <batch_size> <hidden_size> <learning_rate>")
        print("Example: python3 allreduce_worker.py 0 3 2048 128 0.5")
        sys.exit(1)
        
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    hidden_size = int(sys.argv[4])
    learning_rate = float(sys.argv[5])
    
    main(rank, world_size, batch_size, hidden_size, learning_rate)