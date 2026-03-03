// Entry point to test our engine

#include "ag/storage.hpp" // storage class with allocated access
#include <iostream>       // to print

int main() {
  std::cout << "=== Memory Test Starting ===" << std::endl;

  // We create a new "scope" using curly braces. This is like a mini-room in the
  // program.
  {
    std::cout << "Entering the inner scope..." << std::endl;

    // 1. Create a Storage object for 5 integers.
    // sizeof(int) tells us how many bytes 1 integer takes (usually 4 bytes).
    // So 5 * 4 = 20 bytes.
    Storage my_memory(10 * sizeof(int));
    std::cout << "Successfully allocated " << my_memory.size_bytes()
              << " bytes of memory." << std::endl;

    // 2. Get the raw pointer and "cast" it.
    // my_memory.data() returns a void* (a raw bucket of bytes).
    // static_cast<int*> tells the compiler: "Treat this raw bucket as an array
    // of integers."
    int *numbers = static_cast<int *>(my_memory.data());

    // 3. Write bytes to our memory!
    for (int i = 0; i < 10; ++i) {
      numbers[i] = i * 10; // Writing 0, 10, 20, 30, 40
    }

    // 4. Read the bytes back to verify!
    std::cout << "Reading from storage: ";
    for (int i = 0; i < 10; ++i) {
      std::cout << numbers[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "About to leave the inner scope. Watch the memory get "
                 "automatically cleaned up:"
              << std::endl;
  } // <-- The exact moment we hit this brace, my_memory is destroyed!

  std::cout << "Exited the inner scope." << std::endl;
  std::cout << "=== Memory Test Finished ===" << std::endl;

  return 0;
}
