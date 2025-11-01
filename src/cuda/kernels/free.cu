#include "utils.h"

#include <cuda_runtime.h>

// Wrapper function for cudaFree
void cudaFreeWrapper(void* ptr) noexcept {
    // Can't do this due to `noexcept`
    // cudaError_t error = cudaFree(ptr);
    // if (error != cudaSuccess) {
    //     throw std::runtime_error(cudaGetErrorString(error));
    // }
    cudaFree(ptr);
}
