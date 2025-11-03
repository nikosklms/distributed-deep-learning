# ⚙️ Mini Distributed Deep Learning Platform

This project is a custom, from-scratch implementation of a distributed deep learning framework in Python. It uses a **Ring-AllReduce** algorithm to synchronize gradients for data-parallel training, similar to how modern frameworks like **Horovod** or **PyTorch's DDP** operate.

The entire neural network (layers, optimizer, and loss) is built using **only NumPy**.

## Core Features

* **Neural Network Engine from Scratch:** All components are built from the ground up:
    * `Linear` Layer (with He initialization)
    * `ReLU` Activation
    * `CrossEntropyLoss` (with stable Softmax)
    * `SGD` Optimizer
* **Ring-AllReduce Implementation:** A multi-threaded, peer-to-peer (P2P) networking implementation of the Ring-AllReduce algorithm (Reduce-Scatter + All-Gather) to average gradients.
* **Service Discovery:** A simple discovery server allows workers to find each other, register, and form the communication ring.
* **Data-Parallel Training:** The MNIST training dataset is automatically sharded (split) among all participating workers for parallel processing.

## How to Run

This system requires one **Discovery Server** and **N** **Workers**.

### 1. Start the Discovery Server

This script acts as the coordinator to help workers find each other. It must be running first.

```bash
# In your first terminal:
(venv) $ python3 discovery_server.py