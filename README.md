# Distributed Deep Learning Framework with Ring-AllReduce

> **A high‑performance, from‑scratch distributed deep learning framework**

This project is a **custom-built deep learning framework** that implements neural networks, optimizers, and distributed training *without relying on PyTorch or TensorFlow for core logic*.

At its heart is a **decentralized Ring‑AllReduce** backend implemented over **TCP sockets**, accelerated with a **C extension (`fast_net`)** for raw buffer transmission. The framework supports **GPU acceleration via CuPy** and **mixed‑precision (FP16/FP32) training**.

---

## Key Features

* **Custom Neural Network Primitives**
  Hand‑written implementations of:

  * `Linear`, `ReLU`, `CrossEntropy`
  * `Conv2d` (strided `im2col`), `MaxPool2d`
  * `BatchNorm2d`, `Dropout`

* **Decentralized Distributed Training**
  Fully peer‑to‑peer **Ring‑AllReduce** architecture with no parameter server.

* **Hybrid Python / C Networking**
  Performance‑critical socket communication is offloaded to a C extension (`fast_net`) to minimize Python overhead.

* **GPU Acceleration**
  CUDA‑enabled training via **CuPy**, supporting both Linear and CNN workloads.

* **Mixed‑Precision Training**
  FP16 weights & gradients with FP32 master weights and **dynamic loss scaling** for numerical stability.

* **Discovery and Rendezvous**
  A lightweight discovery server initializes workers before switching to direct ring communication.

---

## Project Structure

### Core Components

* `core.py`
  CPU‑based neural network layers (`Linear`, `ReLU`, `CrossEntropy`)

* `core_gpu.py`
  GPU‑accelerated layers and optimizers (`Linear`, `ReLU`, `CrossEntropy`, `AdamW`) using CuPy

* `core_cnn.py`
  Advanced CNN layers (`Conv2d`, `BatchNorm`, `Pooling`) optimized for GPU execution

---

### Training Scripts (Workers)

* `worker_cnn.py`
  Distributed CNN training on **CIFAR‑10** using **AdamW** (GPU)

* `allreduce_worker_gpu.py`
  Distributed **MNIST** training with Linear models (GPU, SGD)

* `allreduce_worker.py`
  CPU‑based MNIST worker for debugging and validation

---

### Distributed Infrastructure

* `discovery_server.py`
  Rendezvous server for worker discovery and ring formation

* `distributed.py`
  Implements the `RingAllReducer` synchronization logic

* `fast_sockets.c`
  C extension for zero‑copy socket transmission of NumPy / CuPy buffers

* `setup.py`
  Build script for the C extension

---

### Data Utilities

* `download_cifar.py`, `load_cifar.py`
  CIFAR‑10 download and preprocessing

* `load_data.py`
  MNIST loading (via `scikit‑learn`)

---

## Installation and Setup

### 1. Prerequisites

* Python **3.8+**
* CUDA‑enabled GPU (recommended)

Install dependencies:

```bash
pip install numpy cupy-cuda12x matplotlib scikit-learn
```

> 🔧 Replace `cupy-cuda12x` with the version matching your CUDA installation (e.g. `cupy-cuda11x`).

---

### 2. Compile the C Extension

The communication backend depends on a compiled C module:

```bash
python3 setup.py build_ext --inplace
```

This generates a shared object such as:

```
fast_net.cpython-3x-x86_64-linux-gnu.so
```

---

### 3. Prepare Datasets

#### CIFAR‑10

```bash
python3 download_cifar.py
```

#### MNIST

```bash
python3 load_data.py
```

(MNIST downloads automatically, but running once caches the data.)

---

## How to Run

The system requires **one discovery server** and **N workers**.

---

### Step 1: Start the Discovery Server

```bash
python3 discovery_server.py
```

Keep this running in a separate terminal.

---

### Step 2: Start Workers

Workers can run on the same machine (testing) or across multiple machines.

---

## Training Scenarios

### Scenario A: CNN on CIFAR‑10 (GPU)

**Command Format:**

```bash
python3 worker_cnn.py <rank> <world_size> <batch_size> <hidden_size> <lr> <epochs>
```

**Example: 2‑Worker Training**

Terminal 1:

```bash
python3 worker_cnn.py 0 2 128 256 0.001 10
```

Terminal 2:

```bash
python3 worker_cnn.py 1 2 128 256 0.001 10
```

---

### Scenario B: Linear Model on MNIST (GPU)

```bash
python3 allreduce_worker_gpu.py <rank> <world_size> <batch_size> <hidden_size> <lr>
```

**Example:**

Terminal 1:

```bash
python3 allreduce_worker_gpu.py 0 2 2048 128 0.5
```

Terminal 2:

```bash
python3 allreduce_worker_gpu.py 1 2 2048 128 0.5
```

---

### Scenario C: Linear Model on MNIST (CPU)

Useful for debugging communication logic.

```bash
python3 allreduce_worker.py <rank> <world_size>
```

**Example: 3 Workers**

```bash
python3 allreduce_worker.py 0 3
python3 allreduce_worker.py 1 3
python3 allreduce_worker.py 2 3
```

---

## Technical Details

### Ring-AllReduce Algorithm

The synchronization proceeds in **two phases**:

1. **Scatter‑Reduce**

   * Gradients are split into *N chunks* (N = number of workers)
   * Each worker sends and accumulates chunks from its neighbor
   * After *N−1 steps*, each worker owns one fully reduced chunk

2. **All‑Gather**

   * Reduced chunks are circulated again
   * After *N−1 steps*, all workers reconstruct the full averaged gradient

---

### Networking Optimization

The `fast_net` C module:

* Accepts raw NumPy / CuPy buffers
* Performs direct `send/recv` operations
* Avoids Python‑level loops and buffer concatenation

This significantly reduces latency in the critical communication path.

---

### Mixed-Precision Strategy (`worker_cnn.py`)

* **Storage:** FP16 weights and activations
* **Master Weights:** FP32 copy for optimizer updates
* **Loss Scaling:**

  * Initial scale: `2^16`
  * Halved on NaN/Inf detection
  * Doubled after 2000 stable iterations

This balances **speed**, **memory efficiency**, and **numerical stability**.

---

## Summary

This project demonstrates how modern distributed deep learning systems can be built **from first principles**, combining:

* Custom autograd‑free neural networks
* Efficient GPU computation
* Low‑level networking
* Decentralized synchronization algorithms

Ideal for **learning**, **research**, and **systems‑level experimentation** in distributed ML.
