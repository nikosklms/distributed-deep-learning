# ⚙️ Mini Distributed Deep Learning Platform

This is a project to build a small-scale distributed deep learning framework from scratch in Python, using only NumPy for the core logic.

The goal is to build a mini-version of PyTorch's `DistributedDataParallel` or Horovod to understand the core principles of distributed training.

## Features / Roadmap

- [ ] **Phase 0: The Core (NumPy NN)**
  - [ ] Build a simple neural network library using only NumPy.
  - [ ] Implement `Linear` and `ReLU` layers (`forward`/`backward`).
  - [ ] Implement `CrossEntropyLoss` and `SGD` optimizer.
  - [ ] Train successfully on MNIST on a single machine.

- [ ] **Phase 1: The Network Pipe (Sockets/gRPC)**
  - [ ] Send NumPy arrays between a client and server.

- [ ] **Phase 2: V1.0 - Parameter Server**
  - [ ] Implement training with one central parameter server and multiple workers.

- [ ] **Phase 3: V2.0 - AllReduce (Ring-AllReduce)**
  - [ ] Implement a decentralized training backend (Ring-AllReduce).

- [ ] **Phase 4: Containerize**
  - [ ] Use Docker and Docker Compose to launch the cluster.
