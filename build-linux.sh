#!/bin/bash

# Check if required arguments are provided
if [ -z "$1" ]; then
    echo "Error: Missing CUDA Toolkit path"
    echo "Usage: ./build-linux.sh CUDA_PATH CMAKE_PREFIX_PATH [delete]"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: Missing CMAKE_PREFIX_PATH"
    echo "Usage: ./build-linux.sh CUDA_PATH CMAKE_PREFIX_PATH [delete]"
    exit 1
fi

CUDA_PATH="$1"
CMAKE_PREFIX_PATH="$2"
DELETE_BUILD=0

# Check if we should delete the build directory
if [ "$3" = "delete" ]; then
    DELETE_BUILD=1
fi

# Check if build directory exists
if [ -d "build" ]; then
    if [ $DELETE_BUILD -eq 1 ]; then
        echo "Removing existing build directory..."
        rm -rf build
        if [ $? -ne 0 ]; then
            echo "Failed to remove build directory."
            exit 1
        fi
        echo "Build directory removed."
    else
        echo "ERROR: Build directory exists! Add 'delete' parameter to remove it."
        echo "Example: ./build-linux.sh CUDA_PATH CMAKE_PREFIX_PATH delete"
        exit 1
    fi
else
    echo "No existing build directory."
fi

# Create build directory
echo "Creating build directory..."
mkdir -p build
if [ $? -ne 0 ]; then
    echo "Failed to create build directory."
    exit 1
fi

# Run CMake configure with provided arguments
echo "Running CMake configuration..."
cmake --preset linux-release -DCUDAToolkit_ROOT="${CUDA_PATH}" -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"
if [ $? -ne 0 ]; then
    echo "CMake configuration failed."
    exit 1
fi

# Change to the build directory
echo "Changing to build directory..."
cd ./build/linux-release/
if [ $? -ne 0 ]; then
    echo "Failed to change directory."
    exit 1
fi

# Build the project
echo "Building project..."
cmake --build . --config Release
if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
fi

echo "Build completed successfully."
exit 0
