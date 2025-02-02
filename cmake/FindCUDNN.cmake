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
    # First check for user-specified path
    if(NOT CUDNN_ROOT_DIR)
        # Check environment variables
        if(DEFINED ENV{CUDNN_ROOT_DIR})
            set(CUDNN_ROOT_DIR $ENV{CUDNN_ROOT_DIR})
        elseif(DEFINED ENV{CUDNN_ROOT})
            set(CUDNN_ROOT_DIR $ENV{CUDNN_ROOT})
        elseif(DEFINED ENV{CUDNN_PATH})
            set(CUDNN_ROOT_DIR $ENV{CUDNN_PATH})
        else()
            # Default installation path on Windows
            set(CUDNN_ROOT_DIR "C:/Program Files/NVIDIA/CUDNN")
        endif()
    endif()
    
    message(STATUS "FindCUDNN: CUDNN_ROOT_DIR = ${CUDNN_ROOT_DIR}")
    
    # Find the latest cuDNN version installed
    file(GLOB _CUDNN_VERSIONS "${CUDNN_ROOT_DIR}/v*")
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
            HINTS 
                "${_LIB_PATH}"
                "${CUDNN_ROOT_DIR}/lib/x64"  # Allow for non-versioned paths too
            NO_DEFAULT_PATH
        )
        
        find_path(CUDNN_INCLUDE_DIR
            cudnn.h
            HINTS 
                "${_INCLUDE_PATH}"
                "${CUDNN_ROOT_DIR}/include"  # Allow for non-versioned paths too
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
