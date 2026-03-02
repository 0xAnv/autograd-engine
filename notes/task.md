# Autograd Engine Project Tasks

## Phase 1: Planning & Setup
- [x] Initial Architecture Planning
- [x] Define Implementation Plan
- [/] Project directory setup and CMake configuration

## Phase 2: Core Tensor & Memory
- [ ] Implement `Storage` and `Allocator` concepts
- [ ] Implement foundational `Tensor` struct (CPU only initially)
- [ ] Implement strides and views abstraction

## Phase 3: CPU Autograd Engine
- [ ] Implement Graph Context and Node logic (`Backward` pass mechanism)
- [ ] Implement basic math operations (Add, Mul) and their derivatives
- [ ] Implement matrix multiplication (CPU naive)
- [ ] Test reverse-mode AD with a simple computational graph

## Phase 4: CUDA Integration
- [ ] Setup CMake for CUDA + C++ compilation separation
- [ ] Implement GPU Memory Allocator
- [ ] Implement basic element-wise CUDA Kernels
- [ ] Implement or Wrap `cuBLAS` for Matrix Multiplication
- [ ] Implement Reduction kernels (Softmax, etc.)

## Phase 5: Neural Network Modules
- [ ] Create `nn::Module` base abstraction
- [ ] Implement `Linear`, `LayerNorm`
- [ ] Implement Optimizers (AdamW)

## Phase 6: Transformer & Training
- [ ] Implement Multi-Head Attention
- [ ] Integrate components into a full Transformer Block
- [ ] Write training loop and data loader for a toy dataset
- [ ] Profile and Optimize VRAM usage for the 3060 (12GB)
