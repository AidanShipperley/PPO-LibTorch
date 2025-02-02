#include <cudnn.h>
#include <iostream>

int main() {
    size_t version = CUDNN_VERSION;
    std::cout << int(version);
    return 0;
}
