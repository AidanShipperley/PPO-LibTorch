# CUDASetup.cmake

# Check for minimum required CUDA version
set(MINIMUM_CUDA_VERSION 12.4)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

#   When using Visual Studio 2022 version >=17.10, the MSVC compiler's version code (_MSC_VER) is updated
#   (in this case, 1940) so that older CUDA toolkits (such as CUDA 12.0 or below) trigger a compatibility
#   error. This happens because the CUDA header file (specifically, host_config.h) includes a check that
#   only permits certain ranges of _MSC_VER values, and versions 12.0 and earlier do not recognize the newer
#   MSVC version. As a result, projects, in particular those using LibTorch C++ code with CUDA, must either
#   upgrade to CUDA 12.4 (or later) where these checks have been updated to support the new MSVC, apply a
#   workaround like the `-allow-unsupported-compiler` flag, or manually patch the header file.
#
#   These alternatives are generally only recommended as a last resort because they might expose the build
#   to unforeseen issues during compilation or runtime. The recommended approach is to update CUDA to a version
#   that officially supports the new MSVC version (in this case, CUDA 12.4 or later) to ensure compatibility
#   and stability without having to adjust driver requirements or risk potential errors.

if(CUDAToolkit_VERSION VERSION_LESS ${MINIMUM_CUDA_VERSION})
    message(FATAL_ERROR 
        "CUDA version ${CUDAToolkit_VERSION} found, but version ${MINIMUM_CUDA_VERSION} or newer is required.\n"
        "This is needed for compatibility with the latest MSVC compilers.\n"
        "Please upgrade CUDA Toolkit or use an older MSVC version."
    )
endif()

message(STATUS "Found CUDA ${CUDAToolkit_VERSION}")

# Set CUDA architectures if not already set from command line
# all - Compile for all known architectures
# all-major - Compile for all major architectures (50, 60, 70, etc.)
# native - Compile for only the current machine's architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Enable search for cuDNN
set(CAFFE2_USE_CUDNN TRUE CACHE BOOL "Use cuDNN" FORCE)

# Set CUDA policies
#if(POLICY CMP0104)
#    cmake_policy(SET CMP0104 NEW)
#endif()