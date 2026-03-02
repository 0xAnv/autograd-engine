// Device that can run our tensor operations

#include <cstdlib>
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
  void *allocate(size_t bytes) override { return std::malloc(bytes); }

  void deallocate(void *ptr, size_t /*bytes*/) override { std::free(ptr); }
};

// Free Function which returns an Allocator Object
Allocator *get_allocator(Device device);