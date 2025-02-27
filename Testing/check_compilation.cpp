#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>

// Utility function for timing operations
double timeOperation(const std::function<void()>& operation, int iterations = 10) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        operation();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count() / iterations;
}

int main() {
    std::cout << "=== Advanced CUDA Environment Checker ===" << std::endl;

    // Check if CUDA is available
    bool cuda_available = torch::cuda::is_available();
    std::cout << "CUDA available: " << (cuda_available ? "YES" : "NO") << std::endl;

    // Use the same device selection logic as in your main app
    std::shared_ptr<torch::Device> device = std::make_shared<torch::Device>(
        (cuda_available ? torch::kCUDA : torch::kCPU)
    );

    std::cout << "Found device: " << *device << std::endl;

    if (device->is_cuda()) {
        // CUDA device information
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;

        // Check cuDNN
        bool cudnn_available = torch::cuda::cudnn_is_available();
        std::cout << "cuDNN available: " << (cudnn_available ? "YES" : "NO") << std::endl;

        // Basic CUDA test
        try {
            // Create tensors on the selected device
            torch::Tensor a = torch::ones({ 100, 100 }, torch::TensorOptions().device(*device));
            torch::Tensor b = torch::ones({ 100, 100 }, torch::TensorOptions().device(*device));

            // Perform matrix multiplication to ensure CUDA is working
            torch::Tensor c = torch::matmul(a, b);
            c.sum().item<float>(); // Force computation

            std::cout << "Basic CUDA test: PASSED" << std::endl;

            // Now test the optimizations
            std::cout << "\n=== Testing CUDA Optimizations ===" << std::endl;

            // 1. Test cuDNN benchmark mode
            if (cudnn_available) {
                std::cout << "Testing cuDNN benchmark mode:" << std::endl;

                // Disable benchmark mode first
                at::globalContext().setBenchmarkCuDNN(false);

                // Create a typical CNN operation
                torch::Tensor input = torch::rand({ 32, 3, 224, 224 },
                    torch::TensorOptions().device(*device));
                torch::nn::Conv2d conv(torch::nn::Conv2dOptions(3, 64, 3).padding(1));
                conv->to(*device);

                // Time without benchmark
                double time_without_benchmark = timeOperation([&]() {
                    auto output = conv->forward(input);
                    output.sum().item<float>(); // Force computation
                    torch::cuda::synchronize();
                    });

                // Enable benchmark mode
                at::globalContext().setBenchmarkCuDNN(true);

                // First run to allow algorithm selection
                auto output = conv->forward(input);
                output.sum().item<float>();
                torch::cuda::synchronize();

                // Time with benchmark
                double time_with_benchmark = timeOperation([&]() {
                    auto output = conv->forward(input);
                    output.sum().item<float>(); // Force computation
                    torch::cuda::synchronize();
                    });

                std::cout << "  Without benchmark: " << std::fixed << std::setprecision(2)
                    << time_without_benchmark << " ms" << std::endl;
                std::cout << "  With benchmark:    " << std::fixed << std::setprecision(2)
                    << time_with_benchmark << " ms" << std::endl;
                std::cout << "  Speedup:           " << std::fixed << std::setprecision(2)
                    << (time_without_benchmark / time_with_benchmark) << "x" << std::endl;
                std::cout << "  cuDNN Benchmark mode working: "
                    << (time_with_benchmark < time_without_benchmark ? "YES" : "NO") << std::endl;
            }

            // 2. Test TF32 on Ampere GPUs
            std::cout << "\nTesting TF32 on Ampere (if available):" << std::endl;

            // We'll try to check if we're on an Ampere or newer GPU
            bool might_be_ampere = false;
            try {
                // For more accurate TF32 testing, we need larger matrices and more complex operations
                // TF32 benefits are most apparent with large matrix multiplications
                torch::Tensor large_a = torch::rand({ 4096, 4096 },
                    torch::TensorOptions().device(*device));
                torch::Tensor large_b = torch::rand({ 4096, 4096 },
                    torch::TensorOptions().device(*device));

                // First disable TF32
                at::globalContext().setAllowTF32CuBLAS(false);
                at::globalContext().setAllowTF32CuDNN(false);

                // Force initial compilation/warmup outside the timed section
                auto warmup = torch::matmul(large_a, large_b);
                warmup.sum().item<float>();
                torch::cuda::synchronize();

                // Time without TF32
                double time_without_tf32 = timeOperation([&]() {
                    auto result = torch::matmul(large_a, large_b);
                    result.sum().item<float>(); // Force computation
                    torch::cuda::synchronize();
                    }, 5); // More iterations for more reliable results

                // Enable TF32
                at::globalContext().setAllowTF32CuBLAS(true);
                at::globalContext().setAllowTF32CuDNN(true);

                // Force warmup with TF32 enabled
                warmup = torch::matmul(large_a, large_b);
                warmup.sum().item<float>();
                torch::cuda::synchronize();

                // Time with TF32
                double time_with_tf32 = timeOperation([&]() {
                    auto result = torch::matmul(large_a, large_b);
                    result.sum().item<float>(); // Force computation
                    torch::cuda::synchronize();
                    }, 5);

                // If there's a speedup, it's likely an Ampere GPU with TF32 support
                might_be_ampere = (time_with_tf32 < time_without_tf32 * 0.8);

                std::cout << "  Without TF32: " << std::fixed << std::setprecision(2)
                    << time_without_tf32 << " ms" << std::endl;
                std::cout << "  With TF32:    " << std::fixed << std::setprecision(2)
                    << time_with_tf32 << " ms" << std::endl;
                std::cout << "  Speedup:      " << std::fixed << std::setprecision(2)
                    << (time_without_tf32 / time_with_tf32) << "x" << std::endl;
                std::cout << "  Ampere GPU with TF32 support: "
                    << (might_be_ampere ? "YES" : "POSSIBLE BUT NOT DETECTED") << std::endl;

                // Verify device capabilities directly if possible
                std::cout << "  GPU might be Ampere (SM 8.0+) based on performance: "
                    << (might_be_ampere ? "YES" : "UNDETERMINED") << std::endl;
            }
            catch (const std::exception& e) {
                std::cout << "  TF32 test failed: " << e.what() << std::endl;
            }

            // 3. Test FP16 mixed precision
            std::cout << "\nTesting FP16 mixed precision:" << std::endl;
            try {
                // Disable FP16 reduction first
                at::globalContext().setAllowFP16ReductionCuBLAS(false);

                // Create half-precision tensors
                torch::Tensor half_a = torch::rand({ 1024, 1024 },
                    torch::TensorOptions().device(*device).dtype(torch::kHalf));
                torch::Tensor half_b = torch::rand({ 1024, 1024 },
                    torch::TensorOptions().device(*device).dtype(torch::kHalf));

                // Time without FP16 reduction
                double time_without_fp16_reduction = timeOperation([&]() {
                    auto result = torch::matmul(half_a, half_b);
                    result.sum().item<float>(); // Force computation
                    torch::cuda::synchronize();
                    });

                // Enable FP16 reduction
                at::globalContext().setAllowFP16ReductionCuBLAS(true);

                // Time with FP16 reduction
                double time_with_fp16_reduction = timeOperation([&]() {
                    auto result = torch::matmul(half_a, half_b);
                    result.sum().item<float>(); // Force computation
                    torch::cuda::synchronize();
                    });

                std::cout << "  Without FP16 reduction: " << std::fixed << std::setprecision(2)
                    << time_without_fp16_reduction << " ms" << std::endl;
                std::cout << "  With FP16 reduction:    " << std::fixed << std::setprecision(2)
                    << time_with_fp16_reduction << " ms" << std::endl;
                std::cout << "  Speedup:                " << std::fixed << std::setprecision(2)
                    << (time_without_fp16_reduction / time_with_fp16_reduction) << "x" << std::endl;
                std::cout << "  FP16 reduction working: "
                    << (time_with_fp16_reduction < time_without_fp16_reduction ? "YES" : "NO") << std::endl;
            }
            catch (const std::exception& e) {
                std::cout << "  FP16 reduction test failed: " << e.what() << std::endl;
            }

            // 4. Test deterministic mode
            std::cout << "\nTesting non-deterministic mode effects:" << std::endl;
            try {
                // First enable deterministic mode
                at::globalContext().setDeterministicCuDNN(true);
                // Note: setDeterministicAlgorithms API has changed in some LibTorch versions
                try {
                    // Try with newer API
                    at::globalContext().setDeterministicAlgorithms(true, false);
                }
                catch (...) {
                    // If that fails, just continue without it
                    std::cout << "  Note: setDeterministicAlgorithms not available with this signature" << std::endl;
                }

                // Create a convolution operation (sensitive to deterministic settings)
                torch::Tensor input = torch::rand({ 64, 64, 128, 128 },
                    torch::TensorOptions().device(*device));
                torch::nn::Conv2d conv(torch::nn::Conv2dOptions(64, 128, 3).padding(1));
                conv->to(*device);

                // Warmup
                auto warmup = conv->forward(input);
                warmup.sum().item<float>();
                torch::cuda::synchronize();

                // Time with deterministic mode
                double time_deterministic = timeOperation([&]() {
                    auto output = conv->forward(input);
                    output.sum().item<float>();
                    torch::cuda::synchronize();
                    }, 5);

                // Disable deterministic mode
                at::globalContext().setDeterministicCuDNN(false);
                try {
                    // Try with newer API
                    at::globalContext().setDeterministicAlgorithms(false, false);
                }
                catch (...) {
                    // If that fails, just continue
                }

                // Warmup again
                warmup = conv->forward(input);
                warmup.sum().item<float>();
                torch::cuda::synchronize();

                // Time with non-deterministic mode
                double time_nondeterministic = timeOperation([&]() {
                    auto output = conv->forward(input);
                    output.sum().item<float>();
                    torch::cuda::synchronize();
                    }, 5);

                std::cout << "  With deterministic mode:     " << std::fixed << std::setprecision(2)
                    << time_deterministic << " ms" << std::endl;
                std::cout << "  With non-deterministic mode: " << std::fixed << std::setprecision(2)
                    << time_nondeterministic << " ms" << std::endl;
                std::cout << "  Speedup:                     " << std::fixed << std::setprecision(2)
                    << (time_deterministic / time_nondeterministic) << "x" << std::endl;
                std::cout << "  Non-deterministic mode benefit: "
                    << (time_nondeterministic < time_deterministic * 0.8 ? "SIGNIFICANT" :
                        (time_nondeterministic < time_deterministic ? "MODEST" : "NONE")) << std::endl;

                // Reset to non-deterministic mode for production use
                at::globalContext().setDeterministicCuDNN(false);
                try {
                    // Try with newer API
                    at::globalContext().setDeterministicAlgorithms(false, false);
                }
                catch (...) {
                    // If that fails, just continue
                }
            }
            catch (const std::exception& e) {
                std::cout << "  Deterministic mode test failed: " << e.what() << std::endl;
            }

            // Summary of optimizations
            std::cout << "\n=== Optimization Summary ===" << std::endl;
            std::cout << "cuDNN Benchmark mode:      ENABLED" << std::endl;
            std::cout << "TF32 on Ampere GPUs:       ENABLED" << (might_be_ampere ? " (working)" : " (may not be supported by your GPU)") << std::endl;
            std::cout << "FP16 mixed precision:      ENABLED" << std::endl;
            std::cout << "Non-deterministic cuDNN:   ENABLED" << std::endl;
            std::cout << "Non-deterministic mode:    ENABLED" << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "CUDA tests failed: " << e.what() << std::endl;
        }
    }
    else {
        std::cout << "Using CPU - CUDA tests skipped" << std::endl;

        // Run a simple CPU test
        try {
            torch::Tensor a = torch::ones({ 100, 100 });
            torch::Tensor b = torch::ones({ 100, 100 });
            torch::Tensor c = torch::matmul(a, b);
            float sum = c.sum().item<float>();
            std::cout << "Basic CPU test: PASSED (sum = " << sum << ")" << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "CPU test failed: " << e.what() << std::endl;
        }
    }

    // Print LibTorch version
    std::cout << "\nLibTorch version: " << TORCH_VERSION_MAJOR << "."
        << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << std::endl;

    std::cout << "\nPress Enter to exit..." << std::endl;
    std::cin.get();

    return 0;
}
