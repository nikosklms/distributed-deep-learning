# Distributed Deep Learning Framework with Ring-AllReduce

> A custom, high-performance distributed deep learning framework built from scratch.

This project implements neural network primitives (`Linear`, `CNN`, `BatchNorm`, `Dropout`) and optimizers (`SGD`, `AdamW`) without relying on PyTorch or TensorFlow for the core logic.

The distributed backend utilizes a custom **Ring-AllReduce** algorithm implemented over TCP sockets, accelerated by a C-extension (`fast_net`) for raw buffer transmission. The framework supports mixed-precision training (FP16/FP32) on GPUs via CuPy.



## ✨ Key Features

* **Custom Network Primitives:** Hand-written implementations of `Conv2d` (using strided `im2col`), `MaxPool2d`, `BatchNorm2d`, and `Dropout`.
* **Distributed Backend:** A decentralized Ring-AllReduce architecture where workers exchange gradients with neighbors to synchronize the model.
* **Hybrid Python/C Networking:** Critical socket operations are offloaded to a C extension (`fast_net`) to minimize Python overhead during gradient synchronization.
* **GPU Acceleration:** Full support for CUDA operations using **CuPy** for both Linear and CNN models.
* **Mixed Precision Training:** Implements FP16 storage for weights/gradients with FP32 master weights and dynamic loss scaling to maintain numerical stability.
* **Discovery Mechanism:** A central discovery server handles the initial handshake (rendezvous) before workers switch to peer-to-peer ring communication.

---

## 📂 Project Structure

### Core Components
* `core.py`: Basic Neural Network layers (`Linear`, `ReLU`, `CrossEntropy`) for CPU.
* `core_gpu.py`: GPU-accelerated Neural Network layers (`Linear`, `ReLU`, `CrossEntropy`, `AdamW`) using CuPy.
* `core_cnn.py`: Advanced CNN layers (`Conv2d`, `BatchNorm`, `Pooling`) optimized for GPU.

### Training Scripts (Workers)
* `worker_cnn.py`: The main training script for CIFAR-10 using the CNN architecture and AdamW optimizer (GPU).
* `allreduce_worker_gpu.py`: GPU-accelerated worker for MNIST training using Linear models and SGD.
* `allreduce_worker.py`: CPU-based worker for MNIST training using Linear models and SGD.

### Distributed Infrastructure
* `discovery_server.py`: The coordination server. Workers register here to find their neighbors in the ring.
* `distributed.py`: Implements the `RingAllReducer` class, managing the synchronization logic.
* `fast_sockets.c`: C-extension code for direct memory access and socket transmission of Numpy/CuPy arrays.
* `setup.py`: Build script for the C-extension.

### Data
* `load_cifar.py` / `download_cifar.py`: Utilities to download and preprocess the CIFAR-10 dataset.
* `load_data.py`: Utilities for the MNIST dataset.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed. You will need the following libraries:

```bash
pip install numpy cupy-cuda12x matplotlib scikit-learn
```

(Note: Replace cupy-cuda12x with the version matching your CUDA installation, e.g., cupy-cuda11x)
2. Compile the C Extension

The communication layer relies on a C module for performance. You must compile it before running any workers.
```bash
python3 setup.py build_ext --inplace
```
This will generate a shared object file (e.g., `fast_net.cpython-3x-x86_64-linux-gnu.so`) in the root directory.

3. Prepare Datasets

Download the required datasets before starting the training cluster.

For CIFAR-10:
```bash
python3 download_cifar.py
```
For MNIST: The script `load_data.py` will handle downloading automatically via sklearn, but running it once ensures data is cached.
```bash
python3 load_data.py
```

🚀 How to Run

The system requires one Discovery Server to be running at all times. Workers connect to this server to discover their peers, form the ring topology, and then begin training.
Step 1: Start the Discovery Server

Open a terminal and start the server. This acts as the rendezvous point.
```bash
python3 discovery_server.py
```
Leave this running in the background or a separate terminal tab.

Step 2: Start Workers

You can run workers on the same machine (for testing) or across different machines (requires network configuration). The examples below assume a single machine using different terminal windows.
Scenario A: Deep CNN on CIFAR-10 (GPU)

This is the most advanced workload. It runs a Convolutional Neural Network on the GPU using `worker_cnn.py`.

Command Syntax: `python3 worker_cnn.py <rank> <world_size> <batch_size> <hidden_size> <lr> <epochs>`

Example: 2-Worker Distributed Training

Terminal 1 (Worker 0):
```bash
python3 worker_cnn.py 0 2 128 256 0.001 10
```
Terminal 2 (Worker 1):
```bash
python3 worker_cnn.py 1 2 128 256 0.001 10
```

Scenario B: Linear Model on MNIST (GPU)

Use this to test GPU acceleration and synchronization on a simpler dataset (MNIST) without the complexity of CNN layers.

Command Syntax: `python3 allreduce_worker_gpu.py <rank> <world_size> <batch_size> <hidden_size> <learning_rate>`

Example: 2-Worker Distributed Training

Terminal 1:
```bash
python3 allreduce_worker_gpu.py 0 2 2048 128 0.5
```
Terminal 2:
```bash
python3 allreduce_worker_gpu.py 1 2 2048 128 0.5
```

Scenario C: Linear Model on MNIST (CPU)

Use this for basic debugging of the ring communication logic if no GPU is available.

Command Syntax: python3 allreduce_worker.py <rank> <world_size>

Example: 3-Worker Distributed Training

    Terminal 1: python3 allreduce_worker.py 0 3

    Terminal 2: python3 allreduce_worker.py 1 3

    Terminal 3: python3 allreduce_worker.py 2 3

🧠 Technical Details
Ring-AllReduce Implementation

The synchronization happens in two phases:

    Scatter-Reduce: The gradient vector is partitioned into N chunks (where N is the number of workers). Workers pass chunks to their right neighbor and add received values to their local buffer. After N−1 steps, every worker holds a fully reduced (summed) portion of the gradient.

    All-Gather: Workers pass the fully reduced chunks around the ring again. After N−1 steps, all workers have the complete, averaged gradient vector.

Networking Optimization

The fast_net C-module allows the Python application to pass a list of Numpy arrays directly to the C layer. The C code handles the buffer iteration and socket transmission (send/recv), avoiding the overhead of concatenating arrays in Python or looping through Python objects during the critical communication path.
Mixed Precision Strategy (worker_cnn.py)

To optimize memory usage and speed on the GPU:

    Storage: Weights and activations are stored in float16.

    Master Weights: A copy of weights is kept in float32 within the MasterNode class for optimizer updates.

    Loss Scaling: A dynamic scaler starts at 216. If Inf or NaN values are detected in gradients, the step is skipped and the scaler is halved. If gradients are stable for 2000 iterations, the scaler is doubled.
