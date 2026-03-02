# Autograd Engine

A lightweight, high-performance automatic differentiation (autograd) engine written in modern C++20.

## Overview

This project implements a core autograd engine from scratch, featuring dynamic computational graphs, custom tensor storage architecture, and memory allocation management suitable for machine learning algorithms.

## Prerequisites

- **CMake** (v3.20 or higher)
- **C++ Compiler** with modern C++20 support (e.g., GCC, Clang, or MSVC)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/0xAnv/autograd-engine.git
cd autograd-engine
```

### 2. Build the Project

You can build the project using the provided helper script:

```bash
chmod +x build.sh
./build.sh
```

Alternatively, you can manually build it using direct CMake commands:

```bash
# 1. Configure the build
cmake -B build

# 2. Compile the project
cmake --build build
```

### 3. Run the Engine Example

Once compiled, you can run the main test executable:

```bash
./build/ag_main
```

## Project Structure

- `src/` - Core implementations for Tensors, custom Storage components, and Memory Allocators.
- `include/ag/` - Public header files defining the engine's core C++ interface.
- `CMakeLists.txt` - Project configuration for CMake.
- `main.cpp` - Entry point and simple example testing the engine.

## License

This project is open-source and available under the MIT License.
