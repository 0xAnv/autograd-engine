#include "ag/allocator.hh"
#include <stdexcept>

// Global static instance of CPUAllocator.
// "static" here means this variable is only visible within this allocator.cpp
// file.
static CPUAllocator cpu_allocator_instance;

Allocator *get_allocator(Device device) {
  if (device == Device::CPU) {
    // Return the address of our global CPU allocator instance
    return &cpu_allocator_instance;
  } else if (device == Device::GPU) {
    // Throw an exception for now, since GPUAllocator isn't implemented
    throw std::runtime_error("CUDAAllocator is not implemented yet.");
  }

  // Fallback if an unknown device is passed
  throw std::invalid_argument("Unknown device type passed to get_allocator.");
}
