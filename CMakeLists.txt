# CMakeLists.txt : CMake project for PPO

# CMake 3.19 is needed for both LibTorch support and support for build presets
cmake_minimum_required (VERSION 3.19 FATAL_ERROR)

# Allow users to specify the path to LibTorch and enforce providing a path.
set(LIBTORCH_PATH "./../libtorch" CACHE PATH "Path to LibTorch")
get_filename_component(LIBTORCH_ABSOLUTE_PATH "${LIBTORCH_PATH}" ABSOLUTE)
message(STATUS "LibTorch absolute path: ${LIBTORCH_ABSOLUTE_PATH}")

# Set CMAKE_MODULE_PATH to find our custom modules BEFORE other includes 
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
set(CMAKE_MODULE_PATH "${PROJECT_ROOT}/cmake;${CMAKE_MODULE_PATH}")

# Enable Hot Reload for MSVC compilers if supported
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# Configure project first with CXX language
project("PPO" LANGUAGES CXX)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Utils)

# Check if LIBTORCH_PATH is provided
if("${LIBTORCH_PATH}" STREQUAL "")
    message(FATAL_ERROR "Please specify the path to LibTorch using -DLIBTORCH_PATH=/path/to/libtorch when invoking CMake.")
else()
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}")
endif()

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

# Enable CUDA if available
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    
    # Configure CUDA architectures
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    
    # Check for minimum required CUDA version
    find_package(CUDAToolkit REQUIRED)
    set(MINIMUM_CUDA_VERSION 12.4)
    
    if(CUDAToolkit_VERSION VERSION_LESS ${MINIMUM_CUDA_VERSION})
        message(FATAL_ERROR 
            "CUDA version ${CUDAToolkit_VERSION} found, but version ${MINIMUM_CUDA_VERSION} or newer is required.\n"
            "This is needed for compatibility with the latest MSVC compilers.\n"
            "Please upgrade CUDA Toolkit or use an older MSVC version."
        )
    endif()
    
    message(STATUS "Found CUDA ${CUDAToolkit_VERSION}")
endif()

# Enable cuDNN for LibTorch
set(USE_CUDNN 1 CACHE BOOL "Use cuDNN" FORCE)
set(CAFFE2_USE_CUDNN 1 CACHE BOOL "Use cuDNN" FORCE)

# Find required LibTorch components
find_package(Torch REQUIRED)

# Diagnostic messages
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDNN_FOUND: ${CUDNN_FOUND}")
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
    message(STATUS "TORCH_CUDA_LIBRARIES: ${TORCH_CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    if(CUDNN_FOUND)
        message(STATUS "CUDNN_LIBRARIES: ${CUDNN_LIBRARIES}")
    endif()
endif()

# Add source to this project's executable.
add_executable (PPO
    "PPO/Agent.h"
    "PPO/Agent.cpp"
    "PPO/PPO_Discrete.h"
    "PPO/PPO_Discrete.cpp"
    "PPO/PPO_MultiDiscrete.h"
    "PPO/PPO_MultiDiscrete.cpp"
    "Distributions/Categorical.h"
    "Distributions/Categorical.cpp"
    "Distributions/CategoricalMasked.h"
    "Distributions/CategoricalMasked.cpp"
    "Environments/MountainCar.h"
    "Environments/MountainCar.cpp"
    "Environments/CartPole.h"
    "Environments/CartPole.cpp"
    "third_party/tomlplusplus/toml.hpp"
    "Utils/ThreadPool.h"
    "Utils/ThreadPool.cpp"
    "Utils/Utils.h"
    "Utils/Utils.cpp"
    "driver.cpp"
)

# Add the CUDA checker executable
add_executable(CompilationChecker "Testing/check_compilation.cpp")

# Link libraries
if(CMAKE_CUDA_COMPILER AND CUDNN_FOUND)
    # Link the executables with LibTorch and cuDNN
    target_link_libraries(PPO PRIVATE 
        ${TORCH_LIBRARIES}
        ${CUDNN_LIBRARIES}
    )
    target_link_libraries(CompilationChecker PRIVATE
        ${TORCH_LIBRARIES}
        ${CUDNN_LIBRARIES}
    )
else()
    # Link the executables just with LibTorch
    target_link_libraries(PPO PRIVATE ${TORCH_LIBRARIES})
    target_link_libraries(CompilationChecker PRIVATE "${TORCH_LIBRARIES}")
endif()

# Silence LibTorch build warnings that I have no control over
if(MSVC)
    # Disable C4267 warnings (conversion from size_t to smaller type)
    target_compile_options(PPO PRIVATE /wd4267)
    target_compile_options(CompilationChecker PRIVATE /wd4267)
endif()

set_property(TARGET PPO PROPERTY CXX_STANDARD 17)
set_property(TARGET CompilationChecker PROPERTY CXX_STANDARD 17)

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