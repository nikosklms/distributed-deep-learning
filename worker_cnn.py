import sys
import time
import numpy as np
import cupy as cp
import gc  # <--- Απαραίτητο για τον καθαρισμό μνήμης

# --- IMPORTS ---
from distributed import RingAllReducer
from load_cifar import load_cifar10
from core_cnn import Conv2d, MaxPool2d, Flatten
from core_gpu import LinearGPU, ReLUGPU, CrossEntropyLossGPU, SGD_GPU

def main(rank, world_size, batch_size, hidden_size, learning_rate, epochs):
    # --- 1. SETUP DATA ---
    print(f"[Node {rank}] Loading CIFAR-10...")
    X_train_all, y_train_all, X_test, y_test = load_cifar10()
    
    # Ανακάτεμα (Shuffling)
    indices = np.arange(X_train_all.shape[0])
    np.random.seed(42); np.random.shuffle(indices)
    X_train_all = X_train_all[indices]
    y_train_all = y_train_all[indices]

    # Μοιρασιά (Sharding)
    num_samples = X_train_all.shape[0]
    samples_per_node = num_samples // world_size
    start = rank * samples_per_node
    end = start + samples_per_node
    
    X_train = X_train_all[start:end]
    y_train = y_train_all[start:end]
    
    # Διαγραφή του μεγάλου dataset από τη RAM
    del X_train_all, y_train_all
    gc.collect()

    print(f"[Node {rank}] Moving data to GPU...")
    X_train_gpu = cp.asarray(X_train)
    y_train_gpu = cp.asarray(y_train)
    
    # Διαγραφή του local dataset από τη RAM (το έχουμε πλέον VRAM)
    del X_train, y_train
    gc.collect()
    
    num_batches = X_train_gpu.shape[0] // batch_size
    print(f"[Node {rank}] Config: Batch={batch_size}, LR={learning_rate}, Epochs={epochs}")

    # --- 2. BUILD MODEL ---
    model = [
        # Block 1
        Conv2d(3, 32, 3, 1, 1), ReLUGPU(), MaxPool2d(2, 2),
        # Block 2
        Conv2d(32, 64, 3, 1, 1), ReLUGPU(), MaxPool2d(2, 2),
        # Block 3
        Conv2d(64, 128, 3, 1, 1), ReLUGPU(), MaxPool2d(2, 2),
        # Head
        Flatten(),
        LinearGPU(2048, hidden_size), ReLUGPU(),
        LinearGPU(hidden_size, 10)
    ]
    
    loss_fn = CrossEntropyLossGPU()
    optimizer = SGD_GPU(model, learning_rate=learning_rate)
    comm = RingAllReducer(rank, world_size)

    # Helper για Flattening Gradients
    trainable_layers = []
    total_params = 0
    for layer in model:
        if hasattr(layer, 'weights'):
            trainable_layers.append(layer)
            total_params += layer.weights.size + layer.biases.size
            
    print(f"[Node {rank}] Total Trainable Params: {total_params}")

    # --- 3. TRAINING LOOP ---
    print(f"[Node {rank}] Starting Training...")
    
    total_training_start = time.time()
    total_compute_time = 0.0
    total_comm_time = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        # LR Decay
        if epoch == int(epochs*0.5): optimizer.learning_rate *= 0.1
        if epoch == int(epochs*0.8): optimizer.learning_rate *= 0.1
        
        for i in range(num_batches):
            t_comp_start = time.time()
            
            s, e = i*batch_size, (i+1)*batch_size
            X_b, y_b = X_train_gpu[s:e], y_train_gpu[s:e]
            
            # Forward
            out = X_b
            for layer in model: out = layer.forward(out)
            loss = loss_fn.forward(out, y_b)
            epoch_loss += float(loss)
            
            # Backward
            optimizer.zero_grad()
            dout = loss_fn.backward()
            for layer in reversed(model): dout = layer.backward(dout)
            
            # Flatten Gradients
            all_grads = []
            for layer in trainable_layers:
                if layer.d_weights is None: layer.d_weights = cp.zeros_like(layer.weights)
                if layer.d_biases is None: layer.d_biases = cp.zeros_like(layer.biases)
                
                all_grads.append(layer.d_weights.ravel())
                all_grads.append(layer.d_biases.ravel())
            
            flat_grads = cp.concatenate(all_grads)
            
            # Padding
            total_len = flat_grads.size
            padding = (world_size - (total_len % world_size)) % world_size
            if padding > 0:
                flat_grads = cp.pad(flat_grads, (0, padding))
            
            # Split & Move to CPU
            chunks_gpu = cp.split(flat_grads, world_size)
            chunks_cpu = [c.get() for c in chunks_gpu]
            
            t_comp_end = time.time()
            
            # Communication
            res_chunks_cpu = comm.allreduce(chunks_cpu)
            
            t_comm_end = time.time()
            
            # Restore & Update
            res_flat = cp.concatenate([cp.asarray(c) for c in res_chunks_cpu])
            if padding > 0: res_flat = res_flat[:-padding]
            
            offset = 0
            for layer in trainable_layers:
                w_size = layer.d_weights.size
                layer.d_weights = res_flat[offset : offset + w_size].reshape(layer.d_weights.shape)
                offset += w_size
                
                b_size = layer.d_biases.size
                layer.d_biases = res_flat[offset : offset + b_size].reshape(layer.d_biases.shape)
                offset += b_size
            
            optimizer.step()
            
            # Stats
            total_compute_time += (t_comp_end - t_comp_start)
            total_comm_time += (t_comm_end - t_comp_end)
            
            if i % 10 == 0 and rank == 0:
                print(f"\rEpoch {epoch+1}: Batch {i}/{num_batches} | Loss: {float(loss):.4f}", end="")

        epoch_dur = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        print(f"\n[Node {rank}] Epoch {epoch+1} Done. Loss: {avg_loss:.4f} | Time: {epoch_dur:.2f}s")

    total_time = time.time() - total_training_start
    comm.close()

    # --- 4. FINAL STATS & EVALUATION ---
    if rank == 0:
        print("\n" + "="*30)
        print(f"TRAINING FINISHED")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Compute Time: {total_compute_time:.2f}s ({(total_compute_time/total_time)*100:.1f}%)")
        print(f"Comm Time:    {total_comm_time:.2f}s ({(total_comm_time/total_time)*100:.1f}%)")
        print("="*30)
        
        print("\nCleaning up memory before Evaluation...")
        
        # 1. Delete training data & optimizer to free VRAM
        del optimizer
        del comm
        del X_train_gpu
        del y_train_gpu
        
        # 2. Force CuPy/Python GC
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        gc.collect()
        
        print("Evaluating on Test Set...")
        X_test_gpu = cp.asarray(X_test)
        y_test_gpu = cp.asarray(y_test)
        
        # 3. Use small test batch size to avoid OOM
        test_batch_size = 100
        num_test_batches = X_test.shape[0] // test_batch_size
        correct = 0
        
        for i in range(num_test_batches):
            s = i * test_batch_size
            e = (i + 1) * test_batch_size
            
            X_b = X_test_gpu[s:e]
            y_b = y_test_gpu[s:e]
            
            out = X_b
            for layer in model:
                out = layer.forward(out)
                # 4. Clear layer cache immediately to save VRAM
                if hasattr(layer, 'x_cols'): layer.x_cols = None
                if hasattr(layer, 'input'): layer.input = None
            
            preds = cp.argmax(out, axis=1)
            correct += cp.sum(preds == y_b)
            
            if i % 10 == 0:
                print(f"\rTesting: {i}/{num_test_batches}", end="")
            
        acc = float(correct) / X_test.shape[0]
        print(f"\n=== FINAL CIFAR-10 ACCURACY: {acc*100:.2f}% ===")

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python3 worker_cnn.py <rank> <world_size> <batch_size> <hidden_size> <lr> <epochs>")
        sys.exit(1)
    
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]))