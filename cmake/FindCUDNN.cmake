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
    
    # Get CUDA major version for search paths
    set(_CUDA_MAJOR_VERSION "${CUDAToolkit_VERSION_MAJOR}")
    message(STATUS "FindCUDNN: CUDA major version: ${_CUDA_MAJOR_VERSION}")
    
    # Function to find the latest available minor version directory for given major version
    function(find_latest_cuda_minor_dir BASE_DIR MAJOR_VERSION OUTPUT_VAR)
        file(GLOB _VERSION_DIRS "${BASE_DIR}/${MAJOR_VERSION}.*")
        
        # Check if we have any directories
        if(_VERSION_DIRS)
            # Sort directories to get the latest version
            list(SORT _VERSION_DIRS)
            list(GET _VERSION_DIRS -1 _LATEST_VERSION_DIR)
            get_filename_component(_VERSION_NAME "${_LATEST_VERSION_DIR}" NAME)
            set(${OUTPUT_VAR} "${_VERSION_NAME}" PARENT_SCOPE)
            message(STATUS "FindCUDNN: Found latest CUDA version directory: ${_VERSION_NAME}")
        else()
            message(STATUS "FindCUDNN: No CUDA version directories found in ${BASE_DIR}")
            set(${OUTPUT_VAR} "" PARENT_SCOPE)
        endif()
    endfunction()
    
    # Function to search for cuDNN in a specific directory structure
    function(find_cudnn_in_directory BASE_PATH VERSION_PATH)
        # Get LATEST available CUDA minor version for specified major version
        if(EXISTS "${BASE_PATH}/${VERSION_PATH}/bin")
            find_latest_cuda_minor_dir("${BASE_PATH}/${VERSION_PATH}/bin" "${_CUDA_MAJOR_VERSION}" LATEST_BIN_VERSION)
        endif()
        
        if(EXISTS "${BASE_PATH}/${VERSION_PATH}/include")
            find_latest_cuda_minor_dir("${BASE_PATH}/${VERSION_PATH}/include" "${_CUDA_MAJOR_VERSION}" LATEST_INCLUDE_VERSION)
        endif()
        
        if(EXISTS "${BASE_PATH}/${VERSION_PATH}/lib")
            find_latest_cuda_minor_dir("${BASE_PATH}/${VERSION_PATH}/lib" "${_CUDA_MAJOR_VERSION}" LATEST_LIB_VERSION)
        endif()
        
        # Use the found latest version or default to current
        if(LATEST_BIN_VERSION)
            set(_BIN_VERSION "${LATEST_BIN_VERSION}")
        else()
            set(_BIN_VERSION "${_CUDA_MAJOR_VERSION}")
        endif()
        
        if(LATEST_INCLUDE_VERSION)
            set(_INCLUDE_VERSION "${LATEST_INCLUDE_VERSION}")
        else()
            set(_INCLUDE_VERSION "${_CUDA_MAJOR_VERSION}")
        endif()
        
        if(LATEST_LIB_VERSION)
            set(_LIB_VERSION "${LATEST_LIB_VERSION}")
        else()
            set(_LIB_VERSION "${_CUDA_MAJOR_VERSION}")
        endif()
        
        message(STATUS "FindCUDNN: Using bin version: ${_BIN_VERSION}")
        message(STATUS "FindCUDNN: Using include version: ${_INCLUDE_VERSION}")
        message(STATUS "FindCUDNN: Using lib version: ${_LIB_VERSION}")
        
        set(_LIB_PATHS
            # Versioned paths based on detected latest version
            "${BASE_PATH}/${VERSION_PATH}/lib/${_LIB_VERSION}/x64"
            # Other common paths without version
            "${BASE_PATH}/${VERSION_PATH}/lib/x64"
            # ZIP install paths
            "${BASE_PATH}/lib/x64"
            "${BASE_PATH}/lib"
        )
        
        set(_INCLUDE_PATHS
            # Versioned paths based on detected latest version
            "${BASE_PATH}/${VERSION_PATH}/include/${_INCLUDE_VERSION}"
            # Other common paths without version
            "${BASE_PATH}/${VERSION_PATH}/include"
            # ZIP install paths
            "${BASE_PATH}/include"
        )
        
        set(_BIN_PATHS
            # Versioned paths based on detected latest version
            "${BASE_PATH}/${VERSION_PATH}/bin/${_BIN_VERSION}"
            # Other common paths without version
            "${BASE_PATH}/${VERSION_PATH}/bin"
            # ZIP install paths
            "${BASE_PATH}/bin"
        )
        
        # Debug print the search paths
        message(STATUS "FindCUDNN: Searching in lib paths:")
        foreach(path ${_LIB_PATHS})
            message(STATUS "  ${path}")
        endforeach()
        
        message(STATUS "FindCUDNN: Searching in include paths:")
        foreach(path ${_INCLUDE_PATHS})
            message(STATUS "  ${path}")
        endforeach()
        
        message(STATUS "FindCUDNN: Searching in bin paths:")
        foreach(path ${_BIN_PATHS})
            message(STATUS "  ${path}")
        endforeach()
        
        # Find Library
        find_library(CUDNN_LIBRARY
            NAMES cudnn cudnn8 cudnn9
            HINTS ${_LIB_PATHS}
            NO_DEFAULT_PATH
        )
        
        # Find Include
        find_path(CUDNN_INCLUDE_DIR
            cudnn.h
            HINTS ${_INCLUDE_PATHS}
            NO_DEFAULT_PATH
        )
        
        # Find DLLs
        if(EXISTS "${CUDNN_LIBRARY}")
            foreach(_bin_path ${_BIN_PATHS})
                # Report which DLL path we're checking
                message(STATUS "FindCUDNN: Checking for DLLs in: ${_bin_path}")
                file(GLOB _CUDNN_DLLS "${_bin_path}/cudnn*.dll")
                if(_CUDNN_DLLS)
                    get_filename_component(CUDNN_DLL_DIR "${_bin_path}" ABSOLUTE)
                    set(CUDNN_DLL_DIR "${CUDNN_DLL_DIR}" PARENT_SCOPE)
                    message(STATUS "FindCUDNN: Found DLLs in: ${CUDNN_DLL_DIR}")
                    break()
                endif()
            endforeach()
        endif()
        
        # Propagate results to parent scope
        set(CUDNN_LIBRARY "${CUDNN_LIBRARY}" PARENT_SCOPE)
        set(CUDNN_INCLUDE_DIR "${CUDNN_INCLUDE_DIR}" PARENT_SCOPE)
    endfunction()

    # First check for GUI installation with version directory
    file(GLOB _VERSION_DIRS "${CUDNN_ROOT_DIR}/v*")
    if(_VERSION_DIRS)
        # Get latest version if multiple exist
        list(SORT _VERSION_DIRS)
        list(GET _VERSION_DIRS -1 _LATEST_VERSION_DIR)
        get_filename_component(_VERSION_NAME "${_LATEST_VERSION_DIR}" NAME)
        message(STATUS "FindCUDNN: Found version directory: ${_VERSION_NAME}")
        
        # Search in versioned directory
        find_cudnn_in_directory("${CUDNN_ROOT_DIR}" "${_VERSION_NAME}")
    endif()
    
    # If not found in version directory, try direct installation layout
    if(NOT CUDNN_LIBRARY OR NOT CUDNN_INCLUDE_DIR)
        message(STATUS "FindCUDNN: Trying direct installation layout")
        find_cudnn_in_directory("${CUDNN_ROOT_DIR}" "")
    endif()
    
    # If still not found, try in CUDA toolkit directory
    if(NOT CUDNN_LIBRARY OR NOT CUDNN_INCLUDE_DIR)
        if(CUDAToolkit_ROOT)
            message(STATUS "FindCUDNN: Trying CUDA toolkit directory: ${CUDAToolkit_ROOT}")
            find_cudnn_in_directory("${CUDAToolkit_ROOT}" "")
        endif()
    endif()
else()
    # On Linux/Mac, look for the library in standard locations
    find_library(CUDNN_LIBRARY
        NAMES cudnn cudnn8 cudnn9
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
    message(STATUS "Found cuDNN:")
    message(STATUS "  Library: ${CUDNN_LIBRARY}")
    message(STATUS "  Include: ${CUDNN_INCLUDE_DIR}")
    message(STATUS "  Version: ${CUDNN_VERSION}")
    
    # Force enable CAFFE2_USE_CUDNN
    set(CAFFE2_USE_CUDNN TRUE CACHE BOOL "Use cuDNN" FORCE)
endif()

mark_as_advanced(
    CUDNN_LIBRARY
    CUDNN_INCLUDE_DIR
)
