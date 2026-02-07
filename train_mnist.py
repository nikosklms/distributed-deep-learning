#!/usr/bin/env python3
"""
train_mnist.py - MNIST training application with distributed support

Supports both CPU and GPU backends.
Usage:
    python3 train_mnist.py --rank 0 --world_size 2 --backend gpu
"""

import argparse
import sys
import time
import numpy as np
import os

from load_data import load_mnist

def log(rank, msg, **kwargs):
    timestamp = time.strftime("%H:%M:%S")
    var_str = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    print(f"[{timestamp}] [App Rank {rank}] {msg} {var_str}")
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Distributed MNIST Training")
    parser.add_argument("rank", type=int, help="Node rank (0 to world_size-1)")
    parser.add_argument("world_size", type=int, help="Total number of nodes")
    parser.add_argument("batch_size", type=int, help="Batch size per node")
    parser.add_argument("hidden_size", type=int, help="Hidden layer size")
    parser.add_argument("lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--backend", choices=['cpu', 'gpu'], default='cpu', help="Compute backend")
    parser.add_argument("--checkpoint_interval", type=int, default=25, help="Checkpoint interval (iterations)")
    args = parser.parse_args()

    # Configuration
    rank = args.rank
    world_size = args.world_size
    backend = args.backend
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    learning_rate = args.lr
    epochs = args.epochs

    log(rank, f"Starting training", backend=backend, world_size=world_size)

    # Backend selection
    if backend == 'gpu':
        try:
            import cupy as cp
            from core_gpu import LinearGPU, ReLUGPU, CrossEntropyLossGPU, SGD_GPU
            from allreduce_gpu import FaultTolerantRingAllReducer
            xp = cp
            Linear = LinearGPU
            ReLU = ReLUGPU
            CrossEntropyLoss = CrossEntropyLossGPU
            SGD = SGD_GPU
            log(rank, "GPU backend loaded")
        except ImportError as e:
            log(rank, f"GPU backend import failed: {e}")
            sys.exit(1)
    else:
        from core import Linear, ReLU, CrossEntropyLoss, SGD
        from allreduce_cpu import RingAllReducer
        xp = np
        log(rank, "CPU backend loaded")

    # Helper to convert array to CPU numpy (for communication)
    def to_cpu(arr):
        if backend == 'gpu' and hasattr(arr, 'get'):
            return arr.get()
        return arr

    # ========================================================================
    # DATA LOADING
    # ========================================================================
    log(rank, "Loading MNIST data...")
    X_train_all, y_train_all, X_test, y_test = load_mnist()

    # Shuffle and partition
    indices = np.arange(X_train_all.shape[0])
    np.random.seed(42)  # Deterministic shuffle for all nodes
    np.random.shuffle(indices)
    X_train_all = X_train_all[indices]
    y_train_all = y_train_all[indices]

    samples_per_node = X_train_all.shape[0] // world_size
    start_idx = rank * samples_per_node
    end_idx = start_idx + samples_per_node

    X_train_node = X_train_all[start_idx:end_idx]
    y_train_node = y_train_all[start_idx:end_idx]

    # Move to device
    X_train_device = xp.asarray(X_train_node)
    y_train_device = xp.asarray(y_train_node)
    
    num_batches = X_train_node.shape[0] // batch_size
    log(rank, f"Data loaded: {X_train_node.shape[0]} samples, {num_batches} batches")

    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================
    log(rank, "Initializing model...")
    layer1 = Linear(input_size=784, output_size=hidden_size, seed=42)
    activation1 = ReLU()
    layer2 = Linear(input_size=hidden_size, output_size=10, seed=42)
    loss_function = CrossEntropyLoss()

    parameters = [layer1, layer2]
    optimizer = SGD(parameters=parameters, learning_rate=learning_rate)
    log(rank, "Model initialized")

    # ========================================================================
    # STATE MANAGEMENT (For Fault Tolerance)
    # ========================================================================
    def get_model_state():
        return {
            'layer1_w': to_cpu(layer1.weights),
            'layer1_b': to_cpu(layer1.biases),
            'layer2_w': to_cpu(layer2.weights),
            'layer2_b': to_cpu(layer2.biases)
        }

    def set_model_state(state):
        layer1.weights = xp.asarray(state['layer1_w'])
        layer1.biases = xp.asarray(state['layer1_b'])
        layer2.weights = xp.asarray(state['layer2_w'])
        layer2.biases = xp.asarray(state['layer2_b'])
        log(rank, "Model state restored")

    # ========================================================================
    # COMMUNICATOR SETUP
    # ========================================================================
    log(rank, "Connecting to cluster...")
    try:
        if backend == 'gpu':
            comm = FaultTolerantRingAllReducer(
                rank=rank,
                world_size=world_size,
                get_model_state_fn=get_model_state,
                set_model_state_fn=set_model_state,
                checkpoint_interval=args.checkpoint_interval,
                verbose=True
            )
        else:
            comm = RingAllReducer(rank=rank, world_size=world_size, verbose=True)
            
        # Give time for mesh to stabilize
        # time.sleep(2)
        log(rank, "Communicator ready")
    except Exception as e:
        log(rank, f"Failed to initialize communicator: {e}")
        sys.exit(1)

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    # Determine start state (for resume)
    if backend == 'gpu':
        current_global_iter = comm.iteration
    else:
        current_global_iter = 0

    start_epoch = current_global_iter // num_batches
    start_batch_idx = current_global_iter % num_batches

    if start_epoch > 0 or start_batch_idx > 0:
        log(rank, f"Resuming from Epoch {start_epoch+1}, Batch {start_batch_idx+1}")
    
    start_time = time.time()
    total_compute_time = 0.0
    total_comm_time = 0.0
    total_batches_processed = 0

    epoch = start_epoch
    batch_start = start_batch_idx
    
    while epoch < epochs:
        log(rank, f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0

        # Inner loop with retry support
        while batch_start < num_batches:
            if batch_start >= num_batches:
                break

            for i in range(batch_start, num_batches):
                batch_num = i + 1
                
                # Get batch
                idx_start = i * batch_size
                idx_end = (i + 1) * batch_size
                X_batch = X_train_device[idx_start:idx_end]
                y_batch = y_train_device[idx_start:idx_end]

                # Forward
                t0 = time.time()
                out1 = layer1.forward(X_batch)
                out_relu = activation1.forward(out1)
                logits = layer2.forward(out_relu)
                loss = loss_function.forward(logits, y_batch)
                running_loss += loss

                # Backward
                optimizer.zero_grad()
                d_logits = loss_function.backward()
                d_layer2 = layer2.backward(d_logits)
                d_relu = activation1.backward(d_layer2)
                d_layer1 = layer1.backward(d_relu)
                total_compute_time += time.time() - t0

                # Prepare gradients for allreduce
                grads_cpu = [
                    to_cpu(layer1.d_weights),
                    to_cpu(layer1.d_biases),
                    to_cpu(layer2.d_weights),
                    to_cpu(layer2.d_biases)
                ]

                # AllReduce
                t0 = time.time()
                log(rank, f"Batch {batch_num}/{num_batches}: AllReduce (loss={float(loss):.4f})")
                
                if backend == 'gpu':
                    avg_grads, info = comm.allreduce(grads_cpu)
                    total_comm_time += time.time() - t0
                    
                    if info['recovered']:
                        resume_iter = info['resume_iteration']
                        log(rank, f"RECOVERY DETECTED! Resuming from iter {resume_iter}")
                        
                        # Update state variables (like CIFAR)
                        new_epoch = resume_iter // num_batches
                        new_batch = resume_iter % num_batches
                        
                        epoch = new_epoch
                        batch_start = new_batch
                        running_loss = 0.0
                        break # Break inner for-loop to restart while-loop
                else:
                    avg_grads = comm.allreduce(grads_cpu)
                    total_comm_time += time.time() - t0

                # Update weights
                layer1.d_weights = xp.asarray(avg_grads[0])
                layer1.d_biases = xp.asarray(avg_grads[1])
                layer2.d_weights = xp.asarray(avg_grads[2])
                layer2.d_biases = xp.asarray(avg_grads[3])
                optimizer.step()

                # Cleanup
                batch_start = i + 1
                total_batches_processed += 1

            else:
                # For-loop finished naturally
                break
        
        # Check if we broke due to recovery (batch_start < num_batches)
        if batch_start < num_batches:
            continue  # Restart while loop with updated epoch/batch_start
        
        # End of epoch success
        if num_batches > 0:
            avg_loss = running_loss / num_batches
            log(rank, f"Epoch {epoch+1}/{epochs} complete: avg_loss={float(avg_loss):.4f}")
        
        epoch += 1
        batch_start = 0  # Reset for next epoch

    training_time = time.time() - start_time
    log(rank, f"Training complete in {training_time:.2f}s")
    
    # Cleanup
    comm.close()

    # ========================================================================
    # EVALUATION & METRICS (Rank 0 only)
    # ========================================================================
    if rank == 0:
        log(rank, "Evaluating on test set...")
        X_test_device = xp.asarray(X_test)
        y_test_device = xp.asarray(y_test)
        
        out1 = layer1.forward(X_test_device)
        out_relu = activation1.forward(out1)
        logits = layer2.forward(out_relu)
        # Handle cupy/numpy argmax difference
        if backend == 'gpu':
            predictions = xp.argmax(logits, axis=1)
            accuracy = xp.mean(predictions == y_test_device)
            acc_val = float(accuracy)
        else:
            predictions = np.argmax(logits, axis=1)
            accuracy = np.mean(predictions == y_test_device)
            acc_val = float(accuracy)
            
        log(rank, f"Test accuracy: {acc_val*100:.2f}%")

        # Metrics Report
        total_wall_time = training_time
        avg_compute = total_compute_time / total_batches_processed if total_batches_processed > 0 else 0
        avg_comm = total_comm_time / total_batches_processed if total_batches_processed > 0 else 0
        compute_pct = (total_compute_time / total_wall_time) * 100 if total_wall_time > 0 else 0
        comm_pct = (total_comm_time / total_wall_time) * 100 if total_wall_time > 0 else 0
        other_time = total_wall_time - (total_compute_time + total_comm_time)
        other_pct = (other_time / total_wall_time) * 100 if total_wall_time > 0 else 0
        total_samples = total_batches_processed * batch_size
        throughput = total_samples / total_wall_time if total_wall_time > 0 else 0

        print("\n" + "=" * 40)
        print("       PERFORMANCE METRICS REPORT       ")
        print("=" * 40)
        print(f"Total Wall Time:      {total_wall_time:.4f}s")
        print(f"System Throughput:    {throughput:.2f} samples/sec")
        print("-" * 40)
        print(f"  • Computation:      {total_compute_time:.4f}s  ({compute_pct:6.2f}%)")
        print(f"  • Communication:    {total_comm_time:.4f}s  ({comm_pct:6.2f}%)")
        print(f"  • Overhead/Idle:    {other_time:.4f}s  ({other_pct:6.2f}%)")
        print("=" * 40 + "\n")

if __name__ == "__main__":
    main()
