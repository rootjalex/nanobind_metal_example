#include "add.h"

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

template<typename value_t, typename size_t>
__global__ void add_kernel(value_t *__restrict__ r, const value_t *__restrict__ a, const value_t *__restrict__ b, const size_t n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        r[i] = a[i] + b[i];
    }
}

float *gpu_add_f32(const float *x, const float *y, const uint64_t n) {
    float *result;
    CHECK_CUDA( cudaMalloc(&result, n * sizeof(float)) );
    gpu_add_out_f32(x, y, result, n);
}

void gpu_add_out_f32(const float *x, const float *y, float *result, const uint64_t n) {
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;

    int threadsPerBlock = maxThreadsPerBlock;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    add_kernel<float, uint64_t><<<blocksPerGrid, threadsPerBlock>>>(result, x, y, n);

    CHECK_CUDA( cudaGetLastError() );

    return result;
}
