#pragma once

#include "ag/allocator.hh"
#include <cstddef>
#include <memory>

class Storage {
public:
  // Constructor declaration: fetches the correct allocator and allocates the
  // memory. The implementation is inside src/storage.cpp
  Storage(size_t size_bytes, Device device = Device::CPU);

  // Accessors to get the raw memory pointer (It's OK to keep these short
  // one-liners in the header)
  void *data() { return data_.get(); }
  const void *data() const { return data_.get(); }

  // Accessors for metadata
  size_t size_bytes() const { return size_bytes_; }
  Device device() const { return device_; }

private:
  std::shared_ptr<void> data_;
  size_t size_bytes_;
  Device device_;
};
