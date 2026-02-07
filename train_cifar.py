#!/usr/bin/env python3
"""
train_cifar.py - Distributed CIFAR-10 training (GPU only)

Implements backend selection (though currently restricted to GPU due to core_cnn dependency)
and integrates with FaultTolerantRingAllReducer.
"""

import sys
import time
import numpy as np
import argparse
import os
import gc

# 1. Imports
try:
    import cupy as cp
    from load_cifar import load_cifar10
    from core_cnn import Conv2d, MaxPool2d, Flatten, GlobalAvgPool2d, BatchNorm2d, Dropout
    from core_gpu import LinearGPU, ReLUGPU, CrossEntropyLossGPU, AdamW_GPU, CosineAnnealingLR_GPU
except ImportError:
    pass # Will handle backend check in main

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

def main():
    parser = argparse.ArgumentParser(description="Distributed CIFAR-10 Training")
    parser.add_argument("rank", type=int, help="Node rank")
    parser.add_argument("world_size", type=int, help="Total number of nodes")
    parser.add_argument("batch_size", type=int, help="Batch size per node")
    parser.add_argument("hidden_size", type=int, help="Hidden layer size")
    parser.add_argument("lr", type=float, help="Learning rate")
    parser.add_argument("epochs", type=int, help="Number of epochs")
    parser.add_argument("--backend", choices=['cpu', 'gpu'], default='gpu', help="Compute backend")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="Checkpoint interval")
    args = parser.parse_args()

    if args.backend == 'cpu':
        print("Error: CIFAR-10 training requires GPU backend (due to CNN ops)")
        sys.exit(1)

    # Import backend-specific communication lib
    try:
        from allreduce_gpu import FaultTolerantRingAllReducer
    except ImportError as e:
        print(f"Failed to import GPU communication lib: {e}")
        sys.exit(1)

    rank = args.rank
    world_size = args.world_size
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    learning_rate = args.lr
    epochs = args.epochs

    print(f"[Node {rank}] Params: Batch={batch_size}, Hidden={hidden_size}, LR={learning_rate}, Backend={args.backend}")
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

    # Also move test data to GPU for validation
    X_test_gpu = cp.asarray(X_test).astype(cp.float16) / 255.0
    y_test_gpu = cp.asarray(y_test)
    del X_test, y_test; gc.collect()

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
    
    master_params = []
    for i, layer in enumerate(model):
        if hasattr(layer, 'weights'):
            name = "Conv1" if i == 0 else f"Layer_{i}"
            master_params.append(MasterNode(layer, name=name))

    optimizer = AdamW_GPU(master_params, learning_rate=learning_rate, weight_decay=1e-2)
    
    # LR Scheduler: Cosine annealing with 5-epoch warmup
    total_steps = epochs * num_batches
    warmup_steps = 5 * num_batches  # 5 epoch warmup
    scheduler = CosineAnnealingLR_GPU(optimizer, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=1e-6)

    # Fault Tolerance Callbacks
    def get_model_state():
        state = {}
        # Save Master Weights (FP32)
        for i, mp in enumerate(master_params):
            if mp.weights is not None:
                state[f'm_w_{i}'] = mp.weights.get()
            if mp.biases is not None:
                state[f'm_b_{i}'] = mp.biases.get()
        
        # Save BatchNorm statistics (FP32)
        idx = 0
        for layer in model:
            if isinstance(layer, BatchNorm2d):
                state[f'bn_mean_{idx}'] = layer.running_mean.get()
                state[f'bn_var_{idx}'] = layer.running_var.get()
                idx += 1
        return state

    def set_model_state(state):
        # Restore Master Weights
        for i, mp in enumerate(master_params):
            if f'm_w_{i}' in state:
                mp.weights[:] = cp.asarray(state[f'm_w_{i}'])
            if f'm_b_{i}' in state:
                mp.biases[:] = cp.asarray(state[f'm_b_{i}'])
            mp.sync_to_model() # Apply to FP16 layers
            
        # Restore BN Stats
        idx = 0
        for layer in model:
            if isinstance(layer, BatchNorm2d):
                if f'bn_mean_{idx}' in state:
                    layer.running_mean[:] = cp.asarray(state[f'bn_mean_{idx}'])
                    layer.running_var[:] = cp.asarray(state[f'bn_var_{idx}'])
                idx += 1
        print(f"[Node {rank}] Model state restored!")

    comm = FaultTolerantRingAllReducer(
        rank=rank, 
        world_size=world_size, 
        get_model_state_fn=get_model_state,
        set_model_state_fn=set_model_state,
        checkpoint_interval=args.checkpoint_interval,
        verbose=True
    )

    total_params = sum(mp.weights.size + (mp.biases.size if mp.biases is not None else 0) 
                       for mp in master_params if mp.weights is not None)
    print(f"[Node {rank}] Total Trainable Params: {total_params}")

    total_training_start = time.time()
    scaler = 65536.0  
    growth_interval = 2000 
    growth_counter = 0
    
    # Performance tracking
    total_compute_time = 0.0
    total_comm_time = 0.0
    total_batches_processed = 0

    # Determine start epoch based on recovery
    start_global_iter = comm.iteration
    start_epoch = start_global_iter // num_batches
    start_batch_idx = start_global_iter % num_batches
    
    epoch = start_epoch
    
    while epoch < epochs:
        if rank == 0: 
            current_lr = scheduler.get_lr()
            print(f"\n[Epoch {epoch+1} Start] LR = {current_lr:.2e}")

        # Enable training mode
        for layer in model:
            if hasattr(layer, 'training'): layer.training = True

        epoch_start = time.time()
        epoch_loss = 0.0
        valid_batches = 0
        
        # Batch loop with rollback support
        if epoch == start_epoch:
            batch_start = start_batch_idx
        else:
            batch_start = 0

        # Inner loop
        while batch_start < num_batches:
            if batch_start >= num_batches:
                break

            for i in range(batch_start, num_batches):
                # Check point logic handled inside comm.allreduce indirectly via iteration
                
                # Sync FP32 master weights -> FP16 model weights
                for mp in master_params: mp.sync_to_model()

                s, e = i*batch_size, (i+1)*batch_size
                X_b, y_b = X_train_gpu[s:e], y_train_gpu[s:e]
                X_b = augment_batch(cp.copy(X_b))
                
                # Forward & Backward
                t0 = time.time()
                out = X_b
                for layer in model: out = layer.forward(out)
                loss = loss_fn.forward(out, y_b)
                
                optimizer.zero_grad()
                dout = loss_fn.backward()
                dout *= scaler 
                for layer in reversed(model): dout = layer.backward(dout)
                total_compute_time += time.time() - t0
                
                # Check overflow
                local_overflow = False
                for mp in master_params:
                    if mp.accumulate_grads(scaler):
                        local_overflow = True
                
                # Prepare gradients
                all_grads = []
                for mp in master_params:
                    if mp.d_weights is not None: all_grads.append(mp.d_weights.ravel())
                    if mp.d_biases is not None: all_grads.append(mp.d_biases.ravel())
                flat_grads = cp.concatenate(all_grads)
                
                # Pad
                total_len = flat_grads.size
                padding = (world_size - (total_len % world_size)) % world_size
                if padding > 0: flat_grads = cp.pad(flat_grads, (0, padding))
                
                chunks_gpu = cp.split(flat_grads, world_size)
                chunks_cpu = [c.get() for c in chunks_gpu]
                
                # AllReduce with Recovery check
                t0 = time.time()
                print("trying")
                res_chunks_cpu, info = comm.allreduce(chunks_cpu)
                total_comm_time += time.time() - t0
                
                if info['recovered']:
                    resume_iter = info['resume_iteration']
                    print(f"[Node {rank}] RECOVERY DETECTED! Resuming from {resume_iter}")
                    
                    # Update state variables
                    new_epoch = resume_iter // num_batches
                    new_batch = resume_iter % num_batches
                    
                    # Assume we recover within reasonable bounds or restart appropriate epoch
                    epoch = new_epoch
                    batch_start = new_batch
                    
                    # Reset scalers/optimizer state (since we didn't checkpoint them fully)
                    scaler = 65536.0
                    growth_counter = 0
                    
                    # Break inner loop to restart with new state
                    break
                
                # Continue if successful
                res_flat = cp.concatenate([cp.asarray(c) for c in res_chunks_cpu])
                if padding > 0: res_flat = res_flat[:-padding]
                
                # Optimizer Step
                if local_overflow or cp.any(cp.isnan(res_flat)) or cp.any(cp.isinf(res_flat)):
                     scaler /= 2.0
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
                    scheduler.step()  # Update learning rate
                    epoch_loss += float(loss)
                    valid_batches += 1
                    growth_counter += 1
                    if growth_counter >= growth_interval:
                        scaler *= 2.0
                        growth_counter = 0
                    total_batches_processed += 1
                
                print(f"\rEpoch {epoch+1}: Batch {i}/{num_batches} | Loss: {float(loss):.4f} | Scale: {scaler}", end="")

                # Prepare for next iteration
                batch_start = i + 1

            else:
                # If for-loop completed without break (no recovery needed)
                break
        
        # Check if we broke out of inner loop due to recovery
        if batch_start < num_batches:
             # We recovering -> loop 'while epoch < epochs' will reiterate with updated 'epoch'
             continue

        # End of epoch success
        avg_loss = epoch_loss / max(1, valid_batches)
        epoch_time = time.time() - epoch_start
        
        # Evaluate Train & Val accuracy
        for layer in model:
            if hasattr(layer, 'training'): layer.training = False
        
        # Train accuracy (sample subset for speed)
        eval_batch_size = 100
        train_correct = 0
        train_samples = min(5000, X_train_gpu.shape[0])  # Sample 5k for speed
        for i in range(train_samples // eval_batch_size):
            s, e = i * eval_batch_size, (i + 1) * eval_batch_size
            out = X_train_gpu[s:e]
            for layer in model: out = layer.forward(out)
            train_correct += int(cp.sum(cp.argmax(out, axis=1) == y_train_gpu[s:e]))
        train_acc = train_correct / train_samples * 100
        
        # Validation accuracy (full test set)
        val_correct = 0
        for i in range(X_test_gpu.shape[0] // eval_batch_size):
            s, e = i * eval_batch_size, (i + 1) * eval_batch_size
            out = X_test_gpu[s:e]
            for layer in model: out = layer.forward(out)
            val_correct += int(cp.sum(cp.argmax(out, axis=1) == y_test_gpu[s:e]))
        val_acc = val_correct / X_test_gpu.shape[0] * 100
        
        # Re-enable training mode
        for layer in model:
            if hasattr(layer, 'training'): layer.training = True
        
        print(f"\n[Node {rank}] Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.2f}s")
        epoch += 1
        start_batch_idx = 0 # Reset for next epoch

    comm.close()

    if rank == 0:
        print(f"\nTRAINING FINISHED | Total Time: {time.time() - total_training_start:.2f}s")
        
        # Final evaluation
        for layer in model:
             if hasattr(layer, 'training'): layer.training = False

        test_batch_size = 100
        correct = 0
        for i in range(X_test_gpu.shape[0] // test_batch_size):
            s = i * test_batch_size; e = (i + 1) * test_batch_size
            out = X_test_gpu[s:e]
            for layer in model: out = layer.forward(out)
            correct += cp.sum(cp.argmax(out, axis=1) == y_test_gpu[s:e])
        
        print(f"\n=== FINAL ACCURACY: {float(correct) / X_test.shape[0] * 100:.2f}% ===")
        
        # Reports
        total_wall_time = time.time() - total_training_start
        total_samples = total_batches_processed * batch_size
        throughput = total_samples / total_wall_time if total_wall_time > 0 else 0
        compute_pct = (total_compute_time / total_wall_time) * 100
        comm_pct = (total_comm_time / total_wall_time) * 100

        print("\n" + "=" * 40)
        print("       PERFORMANCE METRICS REPORT       ")
        print("=" * 40)
        print(f"Total Wall Time:      {total_wall_time:.4f}s")
        print(f"System Throughput:    {throughput:.2f} samples/sec")
        print("-" * 40)
        print(f"  • Computation:      {total_compute_time:.4f}s  ({compute_pct:6.2f}%)")
        print(f"  • Communication:    {total_comm_time:.4f}s  ({comm_pct:6.2f}%)")
        print("=" * 40 + "\n")

if __name__ == "__main__":
    main()