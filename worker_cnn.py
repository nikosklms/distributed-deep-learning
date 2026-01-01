import sys
import time
import numpy as np
import cupy as cp
import gc

from distributed import RingAllReducer
from load_cifar import load_cifar10
from core_cnn import Conv2d, MaxPool2d, Flatten, GlobalAvgPool2d, BatchNorm2d, Dropout
from core_gpu import LinearGPU, ReLUGPU, CrossEntropyLossGPU, AdamW_GPU

def augment_batch(X_batch):
    """Performs random horizontal flips and translations on the GPU."""
    do_flip = cp.random.rand(X_batch.shape[0]) > 0.5
    X_batch[do_flip] = cp.flip(X_batch[do_flip], axis=3)
    
    dx = int(cp.random.randint(-4, 5)) 
    dy = int(cp.random.randint(-4, 5))
    X_batch = cp.roll(X_batch, shift=(dy, dx), axis=(2, 3))
    
    if dx > 0: X_batch[:, :, :, :dx] = 0.0
    elif dx < 0: X_batch[:, :, :, dx:] = 0.0
    if dy > 0: X_batch[:, :, :dy, :] = 0.0
    elif dy < 0: X_batch[:, :, dy:, :] = 0.0
    
    return X_batch

class MasterNode:
    """Manages FP32 master weights for Mixed Precision training."""
    def __init__(self, layer, name="Layer"):
        self.layer = layer
        self.name = name
        self.weights = None
        self.biases = None
        self.d_weights = None
        self.d_biases = None
        if hasattr(layer, 'weights'):
            self.weights = layer.weights.astype(cp.float32)
            self.d_weights = cp.zeros_like(self.weights)
        if hasattr(layer, 'biases'):
            self.biases = layer.biases.astype(cp.float32)
            self.d_biases = cp.zeros_like(self.biases)

    def sync_to_model(self):
        # Downcast FP32 master weights to FP16 for the forward/backward pass
        if self.weights is not None:
            self.layer.weights[:] = self.weights.astype(cp.float16)
        if self.biases is not None:
            self.layer.biases[:] = self.biases.astype(cp.float16)

    def accumulate_grads(self, scaler):
        # Unscale gradients and check for numerical overflow (Inf/NaN)
        has_overflow = False
        def process_tensor(grad_fp16, shape_ref):
            if cp.any(cp.isinf(grad_fp16)) or cp.any(cp.isnan(grad_fp16)):
                return True, cp.zeros_like(shape_ref)
            return False, grad_fp16.astype(cp.float32) / scaler

        if self.weights is not None:
            is_inf, res = process_tensor(self.layer.d_weights, self.weights)
            self.d_weights = res 
            if is_inf: has_overflow = True
        
        if self.biases is not None:
            is_inf, res = process_tensor(self.layer.d_biases, self.biases)
            self.d_biases = res 
            if is_inf: has_overflow = True   
        return has_overflow

def main(rank, world_size, batch_size, hidden_size, learning_rate, epochs):
    print(f"[Node {rank}] Loading CIFAR-10...")
    X_train_all, y_train_all, X_test, y_test = load_cifar10()
    
    # Shuffle and partition data for distributed training
    indices = np.arange(X_train_all.shape[0])
    np.random.seed(42); np.random.shuffle(indices)
    X_train_all = X_train_all[indices]
    y_train_all = y_train_all[indices]

    num_samples = X_train_all.shape[0]
    samples_per_node = num_samples // world_size
    start = rank * samples_per_node; end = start + samples_per_node
    
    X_train = X_train_all[start:end]; y_train = y_train_all[start:end]
    del X_train_all, y_train_all; gc.collect()

    print(f"[Node {rank}] Moving data to GPU...")
    X_train_gpu = cp.asarray(X_train).astype(cp.float16) / 255.0
    y_train_gpu = cp.asarray(y_train)
    del X_train, y_train; gc.collect()

    num_batches = X_train_gpu.shape[0] // batch_size
    
    model = [
        Conv2d(3, 32, 3, 1, 1), BatchNorm2d(32), ReLUGPU(),
        Conv2d(32, 32, 3, 1, 1), BatchNorm2d(32), ReLUGPU(),
        MaxPool2d(2, 2),
        Dropout(0.1),
        
        Conv2d(32, 64, 3, 1, 1), BatchNorm2d(64), ReLUGPU(),
        Conv2d(64, 64, 3, 1, 1), BatchNorm2d(64), ReLUGPU(),
        MaxPool2d(2, 2),
        Dropout(0.1),
        
        Conv2d(64, 128, 3, 1, 1), BatchNorm2d(128), ReLUGPU(),
        Conv2d(128, 256, 3, 1, 1), BatchNorm2d(256), ReLUGPU(),
        MaxPool2d(2, 2),
        
        GlobalAvgPool2d(),
        LinearGPU(256, hidden_size), ReLUGPU(),
        Dropout(0.1),
        LinearGPU(hidden_size, 10)
    ]
    
    loss_fn = CrossEntropyLossGPU()
    comm = RingAllReducer(rank, world_size)

    master_params = []
    for i, layer in enumerate(model):
        if hasattr(layer, 'weights'):
            name = "Conv1" if i == 0 else f"Layer_{i}"
            master_params.append(MasterNode(layer, name=name))

    optimizer = AdamW_GPU(master_params, learning_rate=learning_rate, weight_decay=1e-2)

    total_params = sum(mp.weights.size + (mp.biases.size if mp.biases is not None else 0) 
                       for mp in master_params if mp.weights is not None)
    print(f"[Node {rank}] Total Trainable Params: {total_params}")
    print(f"[Node {rank}] Starting Training with AdamW & Manual Schedule...")

    total_training_start = time.time()
    scaler = 65536.0  
    growth_interval = 2000 
    growth_counter = 0

    for epoch in range(epochs):
        # Manual Step LR decay
        if epoch < 15: current_lr = 1e-3
        elif epoch < 25: current_lr = 1e-4
        else: current_lr = 1e-5
        
        optimizer.learning_rate = current_lr
        if rank == 0: print(f"\n[Epoch {epoch+1} Start] LR set to {current_lr:.1e}")

        # Enable training mode (essential for BatchNorm statistics and Dropout)
        for layer in model:
            if hasattr(layer, 'training'): layer.training = True

        epoch_start = time.time()
        epoch_loss = 0.0
        valid_batches = 0
        
        for i in range(num_batches):
            # Sync FP32 master weights -> FP16 model weights
            for mp in master_params: mp.sync_to_model()

            s, e = i*batch_size, (i+1)*batch_size
            X_b, y_b = X_train_gpu[s:e], y_train_gpu[s:e]
            X_b = augment_batch(cp.copy(X_b))
            
            # Forward & Backward with Loss Scaling
            out = X_b
            for layer in model: out = layer.forward(out)
            loss = loss_fn.forward(out, y_b)
            
            optimizer.zero_grad()
            dout = loss_fn.backward()
            dout *= scaler 
            for layer in reversed(model): dout = layer.backward(dout)
            
            # Check for local overflow (Inf/NaN) after unscaling
            local_overflow = False
            for mp in master_params:
                if mp.accumulate_grads(scaler):
                    local_overflow = True
            
            # Prepare gradients for Ring AllReduce
            all_grads = []
            for mp in master_params:
                if mp.d_weights is not None: all_grads.append(mp.d_weights.ravel())
                if mp.d_biases is not None: all_grads.append(mp.d_biases.ravel())
            flat_grads = cp.concatenate(all_grads)
            
            # Pad gradients to be divisible by world_size, reduce, then reconstruct
            total_len = flat_grads.size
            padding = (world_size - (total_len % world_size)) % world_size
            if padding > 0: flat_grads = cp.pad(flat_grads, (0, padding))
            
            chunks_gpu = cp.split(flat_grads, world_size)
            chunks_cpu = [c.get() for c in chunks_gpu]
            res_chunks_cpu = comm.allreduce(chunks_cpu)
            res_flat = cp.concatenate([cp.asarray(c) for c in res_chunks_cpu])
            if padding > 0: res_flat = res_flat[:-padding]
            
            # Dynamic Loss Scaling Logic
            if local_overflow or cp.any(cp.isnan(res_flat)) or cp.any(cp.isinf(res_flat)):
                 scaler /= 2.0
                 growth_counter = 0
            else:
                offset = 0
                for mp in master_params:
                    if mp.d_weights is not None:
                        w_size = mp.d_weights.size
                        mp.d_weights = res_flat[offset : offset + w_size].reshape(mp.d_weights.shape)
                        offset += w_size
                    if mp.d_biases is not None:
                        b_size = mp.d_biases.size
                        mp.d_biases = res_flat[offset : offset + b_size].reshape(mp.d_biases.shape)
                        offset += b_size
                
                optimizer.step()
                epoch_loss += float(loss)
                valid_batches += 1
                growth_counter += 1
                if growth_counter >= growth_interval:
                    scaler *= 2.0
                    growth_counter = 0
            
            if i % 10 == 0 and rank == 0:
                print(f"\rEpoch {epoch+1}: Batch {i}/{num_batches} | Loss: {float(loss):.4f} | Scale: {scaler}", end="")

        print(f"\n[Node {rank}] Epoch {epoch+1} Done. Loss: {epoch_loss / max(1, valid_batches):.4f} | Time: {time.time() - epoch_start:.2f}s")

    comm.close()

    if rank == 0:
        print(f"\nTRAINING FINISHED | Total Time: {time.time() - total_training_start:.2f}s")
        
        # Cleanup before inference
        del optimizer, comm, X_train_gpu, y_train_gpu, master_params
        cp.get_default_memory_pool().free_all_blocks(); gc.collect()
        
        print("Evaluating...")
        X_test_gpu = cp.asarray(X_test).astype(cp.float16) / 255.0
        y_test_gpu = cp.asarray(y_test)
        
        # Disable dropout/BN updates for inference
        for layer in model:
             if hasattr(layer, 'training'): layer.training = False

        test_batch_size = 100
        correct = 0
        for i in range(X_test.shape[0] // test_batch_size):
            s = i * test_batch_size; e = (i + 1) * test_batch_size
            out = X_test_gpu[s:e]
            for layer in model: out = layer.forward(out)
            correct += cp.sum(cp.argmax(out, axis=1) == y_test_gpu[s:e])
        
        print(f"\n=== FINAL ACCURACY: {float(correct) / X_test.shape[0] * 100:.2f}% ===")

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python3 worker_cnn.py <rank> <world_size> <batch_size> <hidden_size> <lr> <epochs>")
        sys.exit(1)
    
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]))