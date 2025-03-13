@echo off
setlocal enabledelayedexpansion

:: Check if required arguments are provided
if "%~1"=="" (
    echo Error: Missing CUDA Toolkit path
    echo Usage: %0 ^<CUDA_Toolkit_Path^> ^<CMAKE_PREFIX_PATH^>
    exit /b 1
)

if "%~2"=="" (
    echo Error: Missing CMAKE_PREFIX_PATH
    echo Usage: %0 ^<CUDA_Toolkit_Path^> ^<CMAKE_PREFIX_PATH^>
    exit /b 1
)

set CUDA_PATH=%~1
set CMAKE_PREFIX_PATH=%~2

:: Delete the build directory if it exists
if exist build (
    echo Removing existing build directory...
    rd /s /q build
    if errorlevel 1 (
        echo Failed to remove build directory. Exiting.
        exit /b 1
    )
    echo Build directory removed successfully.
) else (
    echo No existing build directory found.
)

:: Create build directory
echo Creating new build directory...
mkdir build
if errorlevel 1 (
    echo Failed to create build directory. Exiting.
    exit /b 1
)

:: Run CMake configure with provided arguments
echo Configuring project with CMake...
cmake --preset x64-release -DCUDAToolkit_ROOT="%CUDA_PATH%" -DCMAKE_PREFIX_PATH="%CMAKE_PREFIX_PATH%"
if errorlevel 1 (
    echo CMake configuration failed. Exiting.
    exit /b 1
)

:: Change to the build directory
echo Changing to build directory...
cd .\build\x64-release\
if errorlevel 1 (
    echo Failed to change to build directory. Exiting.
    exit /b 1
)

:: Build the project
echo Building project...
cmake --build . --config Release
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo Build completed successfully.
exit /b 0
