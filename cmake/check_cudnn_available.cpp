#include <torch/cuda.h>
#include <iostream>

// In your initialization code:
int main() {
    if (torch::cuda::is_available()) {
        if (torch::cuda::cudnn_is_available()) {
            std::cout << "cuDNN is available" << std::endl;
        }
        else {
            std::cout << "cuDNN is NOT available" << std::endl;
        }
    }
    return 0;
}