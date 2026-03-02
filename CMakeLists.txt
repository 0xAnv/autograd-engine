cmake_minimum_required(VERSION 3.20) 
project(autograd_engine VERSION 1.0 LANGUAGES CXX) 

set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The core engine library 
add_library(ag_core STATIC 
    "src/tensor.cpp"
    "src/storage.cpp"
    "src/allocator.cpp"
) 

target_include_directories( ag_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_options(ag_core PRIVATE 
    -Wall -Wextra -Wpedantic -Werror
)

# The main executable 
add_executable(ag_main main.cpp)

# link the executable to our core library
target_link_libraries(ag_main PRIVATE ag_core) 

# applying same strict warnings to our executable 
target_compile_options(ag_main PRIVATE 
    -Wall -Wextra -Wpedantic -Werror
)