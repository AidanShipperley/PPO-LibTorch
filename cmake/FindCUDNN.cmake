# This file is a modified version of the FindCUDNN.cmake from OpenCV (https://github.com/opencv/opencv/blob/a03b81316782ae30038b288fd3568993fa1e3538/cmake/FindCUDNN.cmake) which uses the Apache License v2.0
# This is compatible with the GPL v3.0 license used for this project according to Apache: https://www.apache.org/licenses/GPL-compatibility.html

# The modifications basically remove the FindCUDA.cmake dependency in OpenCV's version, which is quite large and mainly needed for older CMake versions

#[=======================================================================[.rst:
FindCUDNN
---------

Finds the cuDNN library.

Result Variables
^^^^^^^^^^^^^^^

This will define the following variables:

``CUDNN_FOUND``
  True if cuDNN was found
``CUDNN_INCLUDE_DIRS``
  Location of cudnn.h
``CUDNN_LIBRARIES``
  Location of cudnn library
``CUDNN_VERSION``
  Version of cuDNN found
``CUDNN_DLL_DIR``
  Directory containing cuDNN DLLs (Windows only)
#]=======================================================================]

# Print current search state
message(STATUS "FindCUDNN: Starting cuDNN search")

if(WIN32)
    # On Windows, the development files are in a specific location
    set(_KNOWN_CUDNN_ROOT "C:/Program Files/NVIDIA/CUDNN")
    
    # Find the latest cuDNN version installed
    file(GLOB _CUDNN_VERSIONS "${_KNOWN_CUDNN_ROOT}/v*")
    if(_CUDNN_VERSIONS)
        # Get latest version
        list(SORT _CUDNN_VERSIONS)
        list(GET _CUDNN_VERSIONS -1 _LATEST_CUDNN)
        message(STATUS "FindCUDNN: Found latest cuDNN at ${_LATEST_CUDNN}")
        
        # Look for CUDA version-specific paths
        set(_CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
        set(_LIB_PATH "${_LATEST_CUDNN}/lib/${_CUDA_VERSION}/x64")
        set(_INCLUDE_PATH "${_LATEST_CUDNN}/include/${_CUDA_VERSION}")
        
        # Find development files
        find_library(CUDNN_LIBRARY
            NAMES cudnn
            HINTS "${_LIB_PATH}"
            NO_DEFAULT_PATH
        )
        
        find_path(CUDNN_INCLUDE_DIR
            cudnn.h
            HINTS "${_INCLUDE_PATH}"
            NO_DEFAULT_PATH
        )
        
        if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
            message(STATUS "FindCUDNN: Found cuDNN development files:")
            message(STATUS "  Include: ${CUDNN_INCLUDE_DIR}")
            message(STATUS "  Library: ${CUDNN_LIBRARY}")
            
            # Look for runtime DLLs in LibTorch
            if(LIBTORCH_PATH)
                file(GLOB CUDNN_DLLS 
                    "${LIBTORCH_PATH}/lib/cudnn64_*.dll"
                    "${LIBTORCH_PATH}/lib/cudnn_*64_*.dll"
                )
                if(CUDNN_DLLS)
                    message(STATUS "FindCUDNN: Found cuDNN runtime DLLs in LibTorch:")
                    message(STATUS "  DLLs: ${CUDNN_DLLS}")
                    get_filename_component(CUDNN_DLL_DIR "${LIBTORCH_PATH}/lib" ABSOLUTE)
                endif()
            endif()
        endif()
    endif()
else()
    # On Linux/Mac, look for the library in standard locations
    find_library(CUDNN_LIBRARY
        NAMES cudnn
        HINTS ${CUDAToolkit_ROOT}
        PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64
    )
    
    find_path(CUDNN_INCLUDE_DIR
        cudnn.h
        HINTS ${CUDAToolkit_ROOT}
        PATH_SUFFIXES include cuda/include
    )
endif()

# Extract version from header if we found the include directory
if(CUDNN_INCLUDE_DIR)
    if(EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
        file(READ "${CUDNN_INCLUDE_DIR}/cudnn_version.h" CUDNN_H_CONTENTS)
    else()
        file(READ "${CUDNN_INCLUDE_DIR}/cudnn.h" CUDNN_H_CONTENTS)
    endif()

    string(REGEX MATCH "define CUDNN_MAJOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
    set(CUDNN_VERSION_MAJOR ${CMAKE_MATCH_1} CACHE INTERNAL "")
    string(REGEX MATCH "define CUDNN_MINOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
    set(CUDNN_VERSION_MINOR ${CMAKE_MATCH_1} CACHE INTERNAL "")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
    set(CUDNN_VERSION_PATCH ${CMAKE_MATCH_1} CACHE INTERNAL "")

    set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
    message(STATUS "Found cuDNN version: ${CUDNN_VERSION}")
endif()

# Handle the find_package arguments and set the CUDNN_FOUND variable
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN
    FOUND_VAR CUDNN_FOUND
    REQUIRED_VARS
        CUDNN_LIBRARY
        CUDNN_INCLUDE_DIR
    VERSION_VAR CUDNN_VERSION
)

if(CUDNN_FOUND)
    set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
    set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
    message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
    
    # Force enable CAFFE2_USE_CUDNN
    set(CAFFE2_USE_CUDNN TRUE CACHE BOOL "Use cuDNN" FORCE)
endif()

mark_as_advanced(
    CUDNN_LIBRARY
    CUDNN_INCLUDE_DIR
)
