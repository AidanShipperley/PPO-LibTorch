# PPO-LibTorch <br> [![Ubuntu Build (Torch 2.6.0|CUDA 12.6.3)](https://github.com/AidanShipperley/PPO-LibTorch/actions/workflows/ubuntu-build-260-126.yml/badge.svg)](https://github.com/AidanShipperley/PPO-LibTorch/actions/workflows/ubuntu-build-260-126.yml) [![Windows Build (Torch 2.6.0|CUDA 12.6.3)](https://github.com/AidanShipperley/PPO-LibTorch/actions/workflows/windows-build-260-126.yml/badge.svg)](https://github.com/AidanShipperley/PPO-LibTorch/actions/workflows/windows-build-260-126.yml)
PPO-LibTorch is a fully open-source and robust pure C++ implementation of [Proximal Policy Optimization](https://openai.com/index/openai-baselines-ppo/) converted from the wonderful [ICLR Blog Post by Huang, et al](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) using LibTorch, the official C++ frontend for PyTorch.

# Table of Contents
1. [Features](#features)
2. [Requirements](#requirements)
3. [Build From Source](#build)
    1. [Windows](#windows)
    2. [Linux](#linux)
    3. [Test your Build](#test-your-build)
4. [Running The Example Environment](#running-the-example-environment)
5. [Deploying Your Finished Model](#deploying-your-finished-model)


# Features
PPO-LibTorch offers a few unique features:

1. ***A Custom Threadpool Supports PPO's Vectorized Architecture***
    * Avoids the slowdowns associated with spinning up and down new threads.
    * Allows the use of a single learner that collects samples and learns from multiple environments.
    * Execution of these environments all originates from a single process requiring no IPC.
      
2. ***Fully Customizable and Explained Hyperparameters***
    * Hyperparameters can be customized via a config file, no need to recompile for each change.
    * Hyperparameters are clearly explained on this page's wiki.
      
3. ***Custom Environments***
    * Example Environments are provided and can be copied/modified for your needs.
    * No need for you to modify the existing PPO code.
      
4. ***Readable Implementation***
    * This implementation follows the Python implementation as closely as possible.
    * Almost everything is commented so you can tell what each portion is doing.

# Requirements
1. A Windows or Linux machine
2. [CMake](https://cmake.org/download/) (>=3.19), which is required to build this project
3. [LibTorch](https://pytorch.org/get-started/locally/) (>=2.6.0)
<p>&emsp;&emsp;&emsp;
    <img src="https://user-images.githubusercontent.com/70974869/192141845-f631e6c8-d01e-44b8-8af3-38109f76c645.JPG" width="500">
</p>

4. [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) (>=12.4,<=12.6.3), matching the LibTorch version you chose (optional if you are only using CPU)
5. [cuDNN](https://developer.nvidia.com/cudnn-archive) (>=9.7.0) (optional if you are only using CPU)


> Note: You may select either CPU or CUDA, but CUDA is highly recommended. You must have an NVIDIA GPU in order to use CUDA.

# Build

Below are the steps necessary to build this project on your machine.

## Windows

To build this project on Windows, it is recommended that you have Visual Studio 2019 or later and install C++ build tools through the visual studio installer. This is Microsoft's recommended method of accessing C++ build tools on Windows. This section will detail how to build assuming you have Visual Studio.

1. In order to easily access the Clang executable to compile, type `x64` into the Windows search bar and open up the program labeled `x64 Native Tools Command Prompt for VS 20XX`.

2. Confirm that you can run the following commands without any errors and see version numbers:
    ```bash
    cl.exe
    ```

    ```bash
    ninja --version
    ```

    ```bash
    nvcc --version
    ```

3. Nativagate to a directory and clone the repository:
    ```bash
    F:
    cd "F:\Code\"
    ```

    ```bash
    git clone --recurse-submodules https://github.com/AidanShipperley/PPO_LibTorch.git
    ```

    ```bash
    cd "F:\Code\PPO-LibTorch\"
    ```

4. Configure the x64 Release build by running the following.
    1. Set `-DCUDAToolkit_ROOT` to the root directory of your CUDA Toolkit. On Windows, this should be located at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.x\`
    2. Set `-DCMAKE_PREFIX_PATH` to the root directory of your Libtorch installation. This should be the parent directory containing your `\bin` and `\lib` folders.
    ```bash
    cmake --preset x64-release \
    -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" \
    -DCMAKE_PREFIX_PATH="D:\a\PPO-LibTorch\PPO-LibTorch\libtorch\libtorch"
    ```

5. Navigate into the release build directory:
    ```bash
    cd .\build\x64-release\
    ```

6. Compile the release binary and link the executable:
    ```bash
    cmake --build . --config Release
    ```


## Linux

To build this project on Linux, you can follow these steps.

1. Ensure you have libtbb and the Ninja generator:
    ```bash
    sudo apt update
    sudo apt install libtbb-dev ninja-build
    ```

2. Ensure you have the necessary build tools:
    ```bash
    g++ --version
    ```

    ```bash
    ninja --version
    ```

    ```bash
    nvcc --version
    ```

3. Navigate to a directory and clone the repository:
    ```bash
    git clone --recurse-submodules https://github.com/AidanShipperley/PPO_LibTorch.git
    ```

4. Configure the Linux Release build by running the following.
    1. Set `-DCUDAToolkit_ROOT` to the root directory of your CUDA Toolkit. On Ubuntu, this should be located at `/usr/local/cuda-XX.x/`
    2. Set `-DCMAKE_PREFIX_PATH` to the root directory of your Libtorch installation. This should be the parent directory containing your `/bin` and `/lib64` folders.
    ```bash
    cmake --preset linux-release \
    -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 \
    -DCMAKE_PREFIX_PATH=/home/runner/work/PPO-LibTorch/PPO-LibTorch/libtorch/libtorch
    ```

5. Navigate into the release build directory:
    ```bash
    cd ./build/linux-release/
    ```

6. Compile the release binary and link the executable:
    ```bash
    cmake --build . --config Release
    ```


## Test your Build

Included in the PPO build is a standalone executable that you can use to test that CUDA & cuDNN are working properly. You can run this to both confirm that there were no problems with the installation and see what kind of speedup you can expect on your machine, compare it's speed to others, etc.

On Windows:
```bash
.\CompilationChecker.exe
```

On Linux:
```bash
./CompilationChecker
```

---

Congratulations! You can now move onto running the example environment.

# Running The Example Environment

Out of the box, running the built application will run the example environment which is my C++ recreation of [Cart Pole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/). 

On Windows:
```bash
.\PPO.exe
```

On Linux:
```bash
./PPO
```

TODO: Add Argument Parsing and Experiment Logging

# Deploying Your Finished Model

TODO: Write out how to deploy model. It's not super intuitive so I just need to break each step down.
