# Autograd Engine Implementation Plan

This document outlines the architecture, components, and learning path for building a high-performance C++/CUDA autograd engine from scratch, tailored for training Transformer models on a single GPU (like an RTX 3060 with 12GB VRAM).

## 1. Architecture & Component Breakdown

### 1.1 Tensor Core (The Foundation)
The fundamental data structure representing multi-dimensional arrays (tensors).
- **Storage/Memory Management**: Abstraction for allocating and freeing memory on CPU (RAM) and GPU (VRAM).
- **Metadata**: Shape, strides (for views, transposes without copying data), data type, and device placement (CPU/CUDA).
- **Lifecycle Management**: Ensuring memory is released when tensors go out of scope using C++ RAII (Resource Acquisition Is Initialization).

### 1.2 Computational Graph (The Engine)
The core logic for automatic differentiation (reverse-mode autodiff).
- **Nodes/Contexts**: Tracking the history of operations. Every time a tensor is created from a mathematical operation, it records the operation and its parent tensors.
- **Forward Pass**: Executing the operations eagerly while building the Directed Acyclic Graph (DAG).
- **Backward Pass**: Traversing the DAG in reverse topological order, applying the chain rule to accumulate gradients (`grad` attribute of tensors).

### 1.3 Operations & Dispatcher (The Math)
The mathematical functions underlying the neural network.
- **CPU Fallbacks**: Basic implementations for correctness testing.
- **CUDA Kernels**: Highly optimized GPU kernels to maximize 3060 utilization.
  - *Element-wise Ops*: Add, Mul, ReLU, GELU.
  - *Reductions*: Sum, Mean, Max (needed for Softmax, LayerNorm).
  - *Matrix Multiplication*: The most critical operation; requires integrating `cuBLAS` or writing highly tiled memory-aware kernels.
- **Operator Overloading**: Making C++ syntax natural (e.g., `Tensor c = a + b;`).

### 1.4 Neural Network API (`nn` module)
High-level abstractions equivalent to `torch.nn`.
- **Modules**: Base class for network components. Parameter tracking.
- **Layers**: `Linear`, `LayerNorm`, `Dropout`, `Embedding`.
- **Complex Blocks**: Multi-Head Attention, Transformer Block.

### 1.5 Optimizers & Utilities
- **Optimizers**: Implementations of AdamW and SGD to update tensor weights using the computed gradients.
- **Dataloaders**: Efficient batch generation, potentially multi-threaded to prevent CPU bottlenecking the GPU.

---

## 2. Essential C++ (20/23) Skills Mapping

To maximize your learning and achieve senior-level proficiency, here is a breakdown of the specific modern C++ concepts you will need to master to implement each component of the engine.

### For Component 1.1: Tensor Core
- **`std::unique_ptr` & `std::shared_ptr`**: For managing the underlying raw memory buffer (`Storage`).
- **RAII (Resource Acquisition Is Initialization)**: The golden rule of C++. Building safe wrappers around raw `malloc` and `cudaMalloc` so you never leak VRAM.
- **Move Semantics (`std::move`, Rvalue References, `&&`)**: Crucial for transferring ownership of large Tensor buffers across functions without triggering expensive, accidental memory copies.
- **`std::span` (C++20)**: Modern, safe way to view contiguous memory buffers without passing raw pointers and sizes.
- **PIMPL Idiom (Pointer to Implementation)**: An essential systems design pattern. You'll use this to hide CUDA-specific types from your core C++ headers (`.hpp`), keeping your C++ decoupled from `nvcc` and compilation times low.

### For Component 1.2: Computational Graph
- **Graph Data Structures in C++**: Designing a DAG using `std::vector` of `std::shared_ptr` to operation nodes, managing dependencies safely.
- **Topological Sorting algorithms**: Standard Library algorithms (`<algorithm>`) and custom DFS mapping for the backward pass.
- **`std::weak_ptr`**: Breaking reference cycles in the DAG. If Tensors point to Nodes, and Nodes point to Tensors via `shared_ptr`, you will leak memory without weak pointers.
- **Lambdas and `std::function`**: Storing the backward pass computation (the derivative math) inside the graph nodes to execute later when `.backward()` is called.

### For Component 1.3: Operations & Dispatcher
- **Operator Overloading**: Implementing `operator+`, `operator*`, etc., so users can write native math.
- **Templates & C++20 Concepts**: Writing generic operations that apply to any data type (float, int, half). Using `requires` clauses to restrict template instantiation to valid numeric types.
- **CRTP (Curiously Recurring Template Pattern)**: A senior-level C++ idiom for static polymorphism. This avoids the runtime overhead of virtual functions (`vtable` lookups) when dispatching operations, maximizing performance.
- **Variadic Templates**: Useful for generic kernel dispatching functions that take an arbitrary number of arguments and shape dimensions.

### For Component 1.4: Neural Network API (`nn` module)
- **Virtual Functions & Dynamic Polymorphism**: Base `nn::Module` class with virtual `forward()` methods. Unlike core scalar operations, layers can afford virtual function overhead for maximum flexibility.
- **Metaprogramming / Registry Patterns**: To automatically register and iterate over the "parameters" (weights and biases) of a module for the optimizer, similar to PyTorch's parameter registration.
- **`std::vector` and Container Iterators**: Efficiently traversing and updating parameter lists.

### For Component 1.5: Optimizers & Dataloaders
- **`std::thread`, `std::mutex`, `std::condition_variable`**: Multi-threading for the Dataloader to pre-fetch batches asynchronously while the GPU computes the previous batch.
- **`std::atomic`**: Lock-free counters for tracking iterations or thread-safe reference counting.

---

## 3. Required CUDA Topics to Learn

To max out your 3060 (Ampere architecture), you must understand the GPU execution model.

- **Thread Hierarchy**: Threads, Thread Blocks, Grids, and Warps (groups of 32 threads).
- **Memory Hierarchy**: Global Memory (VRAM), Shared Memory (ultra-fast, per-block cache), and Registers. Access patterns (Memory Coalescing) will dictate your performance more than compute.
- **Reductions**: Writing parallel sums and maxes (vital for Softmax and LayerNorm).
- **cuBLAS / cuDNN integration**: While writing your own Matrix Multiplication kernel is educational, a production engine on a 3060 should link to Nvidia's `cuBLAS` for MatMul (`cublasSgemm`) to achieve true hardware utilization.

---

## 4. Learning Resources

### C++ Resources
- **LearnCpp.com**: The best comprehensive, modern guide if you need a ground-up refresher.
- **"A Tour of C++" (3rd Edition)** by Bjarne Stroustrup: Great for an overview of modern C++.
- **"Effective Modern C++"** by Scott Meyers: For mastering Move Semantics and Smart Pointers (a must for the Autograd graph).
- **CppCon YouTube Channel**: specifically talks by Herb Sutter on RAII and modern memory management.

### CUDA Resources
- **"Programming Massively Parallel Processors"** (4th Edition) by Wen-mei Hwu et al.: The definitive textbook on CUDA.
- **UIUC ECE408 / CS483**: Applied Parallel Programming course videos (available on YouTube).
- **NVIDIA CUDA C++ Programming Guide**: The official documentation; read the chapter on the programming model.

### AI Systems / Autograd Design
- **Micrograd** by Andrej Karpathy (YouTube/GitHub): The absolute best starting point to understand the math of the computational graph. It is written in Python, but you will translate this conceptual model to C++.
- **Deep Learning Systems Course (dlsyscourse.org)**: A phenomenal university course (CMU / UW) that literally walks you through building an autograd engine (called "Needle") from scratch.
- **PyTorch Internals (Edward Z. Yang's blog/talks)**: To see how production engines map Tensors, Storage, and Dispatch.

---

## 5. Verification Plan

- Initially, we will test the CPU autograd engine against simple mathematical functions where we can verify the gradients manually.
- When moving to CUDA, we'll verify the outputs of the custom ops and kernels line-up exactly with the CPU fallbacks (with some small tolerance for floating point discrepancies).
- We'll use profiling tools like `nsys` (NVIDIA Nsight Systems) and `ncu` (NVIDIA Nsight Compute) to verify VRAM limits and ensure we are keeping the 3060 saturated under 12 GB.
- The ultimate verification will be training a small transformer block to memorize a sequence and ensuring the loss drops to zero, and that intermediate states match standard benchmarks.
