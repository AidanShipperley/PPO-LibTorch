#!/bin/bash

# Check if required arguments are provided
if [ -z "$1" ]; then
    echo "Error: Missing CUDA Toolkit path"
    echo "Usage: $0 <CUDA_Toolkit_Path> <CMAKE_PREFIX_PATH>"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: Missing CMAKE_PREFIX_PATH"
    echo "Usage: $0 <CUDA_Toolkit_Path> <CMAKE_PREFIX_PATH>"
    exit 1
fi

CUDA_PATH="$1"
CMAKE_PREFIX_PATH="$2"

# Delete the build directory if it exists
if [ -d "build" ]; then
    echo "Removing existing build directory..."
    rm -rf build
    if [ $? -ne 0 ]; then
        echo "Failed to remove build directory. Exiting."
        exit 1
    fi
    echo "Build directory removed successfully."
else
    echo "No existing build directory found."
fi

# Create build directory
echo "Creating new build directory..."
mkdir -p build
if [ $? -ne 0 ]; then
    echo "Failed to create build directory. Exiting."
    exit 1
fi

# Run CMake configure with provided arguments
echo "Configuring project with CMake..."
cmake --preset linux-release -DCUDAToolkit_ROOT="${CUDA_PATH}" -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"
if [ $? -ne 0 ]; then
    echo "CMake configuration failed. Exiting."
    exit 1
fi

# Change to the build directory
echo "Changing to build directory..."
cd ./build/linux-release/
if [ $? -ne 0 ]; then
    echo "Failed to change to build directory. Exiting."
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
