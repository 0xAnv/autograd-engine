#include "ag/tensor.hpp"
#include "ag/storage.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

void hello_tensor() { std::cout << "Hello from ag_core library!" << std::endl; }

size_t sizeof_dtype(DType dtype) {
  switch (dtype) {
  case DType::Float32:
    return sizeof(float);
  case DType::Int32:
    return sizeof(int32_t);
  default:
    throw std::invalid_argument("Unsupported DType");
  }
}

// implementations of Tensor.hpp

// Constructor
Tensor::Tensor(std::vector<int> shape, DType dtype, Device device)
    : shape_(std::move(shape)), dtype_(dtype),
      storage_(
          [](const std::vector<int> &s) {
            size_t count = 1;
            for (int dim : s)
              count *= dim;
            return count;
          }(shape_)*sizeof_dtype(dtype),
          device) {}

// Factory method: empty()
// This creates a tensor using constructor
// Memory inside Storage block will contain whatever garbage were in RAM
Tensor Tensor::empty(std::vector<int> shape, DType dtype) {
  return Tensor(shape, dtype);
}

// Factory method: zeros()
// This creates a tensor but initialise elements to 0
Tensor Tensor::zeros(std::vector<int> shape, DType dtype) {
  Tensor t = empty(shape, dtype);
  size_t size_bytes = t.numel() * sizeof_dtype(dtype);

  std::memset(t.data_ptr<void>(), 0, size_bytes);
  return t;
}
