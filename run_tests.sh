#!/bin/bash
set -e

# Build directory name
BUILD_DIR="build"

echo "=================================="
echo " Building Autograd Engine tests..."
echo "=================================="

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir "$BUILD_DIR"
fi

# Configure and build
echo "Configuring with CMake..."
cmake -B "$BUILD_DIR" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "Compiling tests..."
cmake --build "$BUILD_DIR" -j$(nproc)

# Copy compile commands for clangd if present
if [ -f "$BUILD_DIR/compile_commands.json" ]; then
    cp "$BUILD_DIR/compile_commands.json" .
fi

echo ""
echo "=================================="
echo " Running tests..."
echo "=================================="
echo ""

# Run the tests
cd "$BUILD_DIR"
ctest --output-on-failure

echo ""
echo "=================================="
echo " All tests passed successfully!"
echo "=================================="
