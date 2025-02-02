#include <torch/cuda.h>

// In your initialization code:
if (torch::cuda::is_available()) {
    if (torch::cuda::cudnn_is_available()) {
        std::cout << "cuDNN is available" << std::endl;
    }
    else {
        std::cout << "cuDNN is NOT available" << std::endl;
    }
}
