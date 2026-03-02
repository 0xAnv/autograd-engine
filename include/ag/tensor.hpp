#pragma once
#include "ag/storage.hpp"
#include <cstddef>
#include <vector>

// data type enum
enum class DType { Float32, Int32 };
size_t sizeof_dtype(DType dtype); // gives size of dtype in bytes

// Tensor class
class Tensor {
public:
  // constructor
  Tensor(std::vector<int> shape, DType dtype = DType::Float32,
         Device device = Device::CPU);

  // Accessors
  const std::vector<int> &shape() const { return shape_; }
  DType dtype() const { return dtype_; }

  // Computes the total number of elements in tensor
  // We compute this on the fly
  int numel() const {
    int count = 1;
    for (int dim : shape_)
      count *= dim;
    return count;
  }

  // Data pointers
  // Templated methods let's the user request a specific pointer type
  // eg. float* instead of dealing with the raw void* from Storage
  template <typename T> T *data_ptr() {
    return static_cast<T *>(storage_.data());
  }

  // constant version for when the Tensor itself is const
  template <typename T> const T *data_ptr() const {
    return static_cast<T *>(storage_.data());
  }

  // Factory methods
  static Tensor empty(std::vector<int> shape, DType dtype = DType::Float32);
  static Tensor zeros(std::vector<int> shape, DType dtype = DType::Float32);

private:
  std::vector<int> shape_;
  DType dtype_;
  Storage storage_;
};