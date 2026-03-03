#include "ag/storage.hpp"

// Storage class Initializer
Storage::Storage(size_t size_bytes, Device device)
    : size_bytes_(size_bytes), device_(device) {

  // Fetch the correct allocator for our device
  Allocator *allocator = get_allocator(device);

  // Ask the allocator for a block of raw memory
  void *raw_ptr = allocator->allocate(size_bytes);

  // Define a "custom deleter". This is a lambda function (an anonymous for our
  // shared pointer to use)
  auto deleter = [allocator, size_bytes](void *ptr) {
    allocator->deallocate(ptr, size_bytes);
  };

  // Wrap the raw pointer in a shared_ptr with our custom deleter.
  data_ = std::shared_ptr<void>(raw_ptr, deleter);
}
