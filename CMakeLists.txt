# CMakeList.txt : CMake project for PPO
cmake_minimum_required (VERSION 3.8)

# Allow users to specify the path to LibTorch and enforce providing a path.
set(LIBTORCH_PATH "./../libtorch" CACHE PATH "Path to LibTorch")
get_filename_component(LIBTORCH_ABSOLUTE_PATH "${LIBTORCH_PATH}" ABSOLUTE)
message(STATUS "LibTorch absolute path: ${LIBTORCH_ABSOLUTE_PATH}")

# Set CMAKE_MODULE_PATH to find our custom modules BEFORE other includes
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
set(CMAKE_MODULE_PATH "${PROJECT_ROOT}/cmake;${CMAKE_MODULE_PATH}")

# Print module path for debugging
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
message(STATUS "Looking for FindCUDNN.cmake in: ${PROJECT_ROOT}/cmake")

include(CheckLanguage)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# Set cuDNN policy to use ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# Set CUDA and cuDNN flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWITH_CUDNN=1")
set(USE_CUDNN 1 CACHE BOOL "Use cuDNN for deep learning primitives" FORCE)

# Attempt to enable cudnn
project ("PPO" LANGUAGES CUDA CXX)

# Check if LIBTORCH_PATH is provided
if("${LIBTORCH_PATH}" STREQUAL "")
    message(FATAL_ERROR "Please specify the path to LibTorch using -DLIBTORCH_PATH=/path/to/libtorch when invoking CMake.")
else()
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}")
endif()

# Find CUDA and cuDNN
find_package(CUDAToolkit REQUIRED)

# Try to find cuDNN with our custom module first
message(STATUS "Attempting to find cuDNN with custom module...")
find_package(CUDNN QUIET)

if(NOT CUDNN_FOUND)
    message(STATUS "Could not find cuDNN with custom module, falling back to system paths")
    # Add any fallback detection here if needed
endif()

# Print cuDNN detection results
message(STATUS "CUDNN detection results:")
message(STATUS "  CUDNN_FOUND: ${CUDNN_FOUND}")
message(STATUS "  CUDNN_INCLUDE_DIRS: ${CUDNN_INCLUDE_DIRS}")
message(STATUS "  CUDNN_LIBRARIES: ${CUDNN_LIBRARIES}")
message(STATUS "  CUDNN_VERSION: ${CUDNN_VERSION}")

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    message(STATUS "CUDAToolkit_ROOT: ${CUDAToolkit_ROOT}")

    find_library(CUDA_nvrtc_LIBRARY nvrtc ${CUDAToolkit_ROOT})
    message(STATUS "CUDA_nvrtc_LIBRARY: ${CUDA_nvrtc_LIBRARY}")
endif()

# Find required LibTorch components
find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})
if(CUDNN_FOUND)
    include_directories(${CUDNN_INCLUDE_DIRS})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Diagnostic messages
message(STATUS "CUDNN_FOUND: ${CUDNN_FOUND}")
message(STATUS "USE_CUDNN: ${USE_CUDNN}")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_CUDA_LIBRARIES: ${TORCH_CUDA_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")

# Add source to this project's executable.
add_executable (PPO 
    "PPO_Discrete.cpp" 
    "PPO_Discrete.h" 
    "PPO_MultiDiscrete.cpp" 
    "PPO_MultiDiscrete.h" 
    "driver.cpp" 
    "ThreadPool.h" 
    "ThreadPool.cpp" 
    "Agent.h" 
    "Agent.cpp" 
    "Distributions/CategoricalMasked.h" 
    "Distributions/CategoricalMasked.cpp" 
    "Environments/MountainCar.h" 
    "Environments/MountainCar.cpp" 
    "Distributions/Categorical.h" 
    "Distributions/Categorical.cpp" 
    "Environments/CartPole.h" 
    "Environments/CartPole.cpp" 
    "Utils.h" 
    "Utils.cpp"
 "cmake/check_cudnn_available.cpp")

# Link libraries
if(CUDNN_FOUND)
    target_link_libraries(PPO PRIVATE 
        ${TORCH_LIBRARIES}
        ${CUDNN_LIBRARIES}
    )
else()
    target_link_libraries(PPO PRIVATE ${TORCH_LIBRARIES})
endif()

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET PPO PROPERTY CXX_STANDARD 17)
endif()

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    if(CUDNN_DLL_DIR)
        file(GLOB CUDNN_DLLS "${CUDNN_DLL_DIR}/cudnn*.dll")
    endif()
    add_custom_command(TARGET PPO
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       ${CUDNN_DLLS}
                       $<TARGET_FILE_DIR:PPO>)
endif (MSVC)

# Set CUDA architecture to 86 for RTX 30 series GPU (as detected in CMake output)
set_target_properties(PPO PROPERTIES CUDA_ARCHITECTURES "86")