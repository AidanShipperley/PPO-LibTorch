@echo off
setlocal EnableDelayedExpansion

:: Check if required arguments are provided
if "%~1"=="" (
    echo Error: Missing CUDA Toolkit path
    echo Usage: .\build-windows.bat CUDA_PATH CMAKE_PREFIX_PATH [delete]
    exit /b 1
)

if "%~2"=="" (
    echo Error: Missing CMAKE_PREFIX_PATH
    echo Usage: .\build-windows.bat CUDA_PATH CMAKE_PREFIX_PATH [delete]
    exit /b 1
)

set CUDA_PATH=%~1
set CMAKE_PREFIX_PATH=%~2
set DELETE_BUILD=0

:: Check if third parameter is 'delete'
if /i "%~3"=="delete" (
    set DELETE_BUILD=1
)

:: Build directory handling
if exist build (
    if !DELETE_BUILD!==1 (
        echo Removing existing build directory...
        rd /s /q build
        if errorlevel 1 (
            echo Failed to remove build directory. Exiting.
            exit /b 1
        )
        echo Build directory removed.
    ) else (
        echo ERROR: Build directory exists! Add 'delete' parameter to remove it.
        echo Example: .\build-windows.bat CUDA_PATH CMAKE_PREFIX_PATH delete
        exit /b 1
    )
) else (
    echo No existing build directory.
)

:: Create build directory
echo Creating build directory...
mkdir build
if errorlevel 1 (
    echo Failed to create build directory.
    exit /b 1
)

:: Run CMake configure
echo Running CMake configuration...
cmake --preset x64-release -DCUDAToolkit_ROOT="%CUDA_PATH%" -DCMAKE_PREFIX_PATH="%CMAKE_PREFIX_PATH%"
if errorlevel 1 (
    echo CMake configuration failed.
    exit /b 1
)

:: Change to build directory
echo Changing to build directory...
cd .\build\x64-release\
if errorlevel 1 (
    echo Failed to change directory.
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
