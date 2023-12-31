﻿# CMakeList.txt : CMake project for PPO, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# Attempt to enable cudnn
project ("PPO" LANGUAGES CUDA CXX)

# Allow users to specify the path to LibTorch and enforce providing a path.
set(LIBTORCH_PATH "./../libtorch" CACHE PATH "Path to LibTorch")
if("${LIBTORCH_PATH}" STREQUAL "")
    message(FATAL_ERROR "Please specify the path to LibTorch using -DLIBTORCH_PATH=/path/to/libtorch when invoking CMake.")
else()
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}")
endif()

# Find required LibTorch components
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "/O2")

# Add source to this project's executable.
add_executable (PPO "PPO_Discrete.cpp" "PPO_Discrete.h" "PPO_MultiDiscrete.cpp" "PPO_MultiDiscrete.h" "driver.cpp" "ThreadPool.h" "ThreadPool.cpp" "Agent.h" "Agent.cpp" "Distributions/CategoricalMasked.h" "Distributions/CategoricalMasked.cpp" "Environments/MountainCar.h" "Environments/MountainCar.cpp" "Distributions/Categorical.h" "Distributions/Categorical.cpp" "Environments/CartPole.h" "Environments/CartPole.cpp" "Utils.h" "Utils.cpp")
target_link_libraries(PPO "${TORCH_LIBRARIES}")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET PPO PROPERTY CXX_STANDARD 20)
endif()

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET PPO
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:PPO>)
endif (MSVC)

# TODO: Add tests and install targets if needed.
