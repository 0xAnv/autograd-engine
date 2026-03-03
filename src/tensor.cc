#include "ag/tensor.hh"
#include "ag/storage.hh"
#include <cstddef>
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

// Helper function to calculate default strides contiguous for a given shape
// Example: Shape [2,3,4] -> strides [12,4,1]
static std::vector<int>
compute_contiguous_strides(const std::vector<int> &shape) {
  std::vector<int> strides(shape.size(), 1);
  if (shape.empty())
    return strides;
  for (int i = (int)shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

// Constructor
Tensor::Tensor(std::vector<int> shape, DType dtype, Device device)
    : shape_(std::move(shape)), strides_(compute_contiguous_strides(shape_)),
      dtype_(dtype), storage_offset_(0), storage_(
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

// Private constructor for creating views
Tensor::Tensor(std::vector<int> shape, std::vector<int> strides,
               size_t storage_offset, DType dtype, Storage storage)
    : shape_(std::move(shape)), strides_(std::move(strides)), dtype_(dtype),
      storage_offset_(storage_offset), storage_(std::move(storage)) {}

bool Tensor::is_contiguous() const {
  std::vector<int> expected_strides = compute_contiguous_strides(shape_);
  return strides_ == expected_strides;
}

int Tensor::stride(int dim) const {
  if (dim < 0 || dim >= static_cast<int>(strides_.size())) {
    throw std::out_of_range("Dimension out of range.");
  }
  return strides_[dim];
}

size_t Tensor::storage_offset() const { return storage_offset_; }

// view reinterprets the shape of the tensor.
// It requires a contiguous storage block
Tensor Tensor::view(std::vector<int> shape) const {
  size_t num_numel = 1;
  for (int d : shape)
    num_numel *= d;
  if (num_numel != static_cast<size_t>(this->numel())) {
    throw std::invalid_argument(
        "view() shape cannot change the total number of elements");
  }

  if (!is_contiguous()) {
    throw std::runtime_error("cannot call view() on a non-contiguous tensor");
  }

  std::vector<int> new_strides = compute_contiguous_strides(shape);

  // Returnsa new tensor sharing the exact same storage block
  return Tensor(std::move(shape), std::move(new_strides), storage_offset_,
                dtype_, storage_);
}

// transpose() swaps two dimensions . Highly efficient: only modifies vectors, 0
// data moves.
Tensor Tensor::transpose(int dim0, int dim1) const {
  if (dim0 < 0 || dim0 >= static_cast<int>(shape_.size()) || dim1 < 0 ||
      dim1 >= static_cast<int>(shape_.size())) {
    throw std::out_of_range("Dimension out of range");
  }

  std::vector<int> new_shape = shape_;
  std::vector<int> new_strides = strides_;

  std::swap(new_shape[dim0], new_shape[dim1]);
  std::swap(new_strides[dim0], new_strides[dim1]);

  return Tensor(std::move(new_shape), std::move(new_strides), storage_offset_,
                dtype_, storage_);
}

// permute() rearranges the dimensions according to `dims`.
Tensor Tensor::permute(std::vector<int> dims) const {
  if (dims.size() != shape_.size()) {
    throw std::invalid_argument(
        "permute: dims must have the same number of elements as tensor's rank");
  }

  std::vector<int> new_shape(shape_.size());
  std::vector<int> new_strides(strides_.size());

  for (size_t i = 0; i < dims.size(); ++i) {
    int d = dims[i];
    if (d < 0 || d >= static_cast<int>(shape_.size())) {
      throw std::out_of_range("Dimension out of range in permute");
    }
    new_shape[i] = shape_[d];
    new_strides[i] = strides_[d];
  }

  return Tensor(std::move(new_shape), std::move(new_strides), storage_offset_,
                dtype_, storage_);
}
