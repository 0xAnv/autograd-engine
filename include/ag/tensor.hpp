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
  const std::vector<int> &strides() const { return strides_; }
  DType dtype() const { return dtype_; }

  bool is_contiguous() const;
  int stride(int dim) const;
  size_t storage_offset() const;

  // View operations
  Tensor view(std::vector<int> shape) const;
  Tensor transpose(int dim0, int dim1) const;
  Tensor permute(std::vector<int> dims) const;

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
  void *data_ptr() {
    return static_cast<char *>(storage_.data()) +
           storage_offset_ * sizeof_dtype(dtype_);
  }

  const void *data_ptr() const {
    return static_cast<const char *>(storage_.data()) +
           storage_offset_ * sizeof_dtype(dtype_);
  }

  template <typename T> T *data_ptr() { return static_cast<T *>(data_ptr()); }

  template <typename T> const T *data_ptr() const {
    return static_cast<const T *>(data_ptr());
  }

  // Factory methods
  static Tensor empty(std::vector<int> shape, DType dtype = DType::Float32);
  static Tensor zeros(std::vector<int> shape, DType dtype = DType::Float32);

private:
  // New private constructor to create views
  Tensor(std::vector<int> shape, std::vector<int> strides,
         size_t storage_offset, DType dtype, Storage storage);
  std::vector<int> shape_;
  std::vector<int> strides_;
  DType dtype_;
  size_t storage_offset_;
  Storage storage_;
};