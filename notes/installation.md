
```bash 
sudo apt update
sudo apt install build-essential cmake ninja-build gdb 
```

We do this because: 
- `build essential` : gives you `gcc` and `g++` 
- `cmake` : standard build system generator for C++ projects  
- `ninja-build` :  A blazing fast build tool that replaces `make`. It significantly speeds up compiling large C++ projects.
- `gdb` : The GNU debugger (crucial for finding segfaults when dealing with raw memory later).

### PRoject setups and cmake configs 

- In c++ world, CMake is the industry standard build system. 
- We use it because it makes it trivial to manage dependencies (like linking to CUDA later), configure compiler flags (like enabling C++20 standards and turning on strict warnings) while structuring our project cleanly. 

```text
autograd-engine/
├── CMakeLists.txt        # The root build script
├── include/
│   └── ag/               # 'ag' namespace. All public headers (.hpp) go here.
├── src/                  # All private implementation files (.cpp) go here.
└── main.cpp              # The entry point for testing our engine.

```

What CMakeLists.txt should have : 
- It must require C++20 standard 
- It should define a library target called `ag_core` (which will hold tensor and autograd code)
- It should define an executable target called `ag_main` (built from `main.cpp`) 
- `ag_main` should link to `ag_core` 
- Add strict compiler warnings to executables (eg. -Wall -Wextra -Wpedantic -Werror)

