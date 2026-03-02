cmake_minimum_required(VERSION 3.20) 
cmake_policy(SET CMP0135 NEW) # new timestamping policy for downloads
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


# Testing with GoogleTest framework 
include(FetchContent)
FetchContent_Declare(
    googletest 
    URL 
    https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: prevent over riding the parent project's compiler/linked settings 
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) 
FetchContent_MakeAvailable(googletest) 

# Enable testing in CMake 
enable_testing() 

# Create out test executable 
add_executable(
    ag_test 
    "tests/test_tensor.cpp"
)

# Link our core engine and the GTest libraries 
target_link_libraries(
    ag_test 
    PRIVATE ag_core 
    GTest::gtest_main
)

# Register the executable with CTest (CMake's test runner) 
include(GoogleTest)
gtest_discover_tests(ag_test)