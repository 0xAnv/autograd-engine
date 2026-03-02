#include <cstdlib>
#include <new>
enum class Device { CPU, GPU };

// Allocator base class -> Allocates memory
class Allocator {
public:
  virtual ~Allocator() = default;
  virtual void *allocate(size_t bytes) = 0;
  virtual void deallocate(void *ptr, size_t bytes) = 0;
};

// CPU Allocator
class CPUAllocator : public Allocator {
public:
  void *allocate(size_t bytes) override {
    // return std::malloc(bytes);
    void *ptr = std::malloc(bytes);
    if (ptr == nullptr && bytes > 0) { // check for nullptr
      throw std::bad_alloc();
    }
    return ptr;
  }

  void deallocate(void *ptr, size_t /*bytes*/) override { std::free(ptr); }
};

// Free Function which returns an Allocator Object
Allocator *get_allocator(Device device);