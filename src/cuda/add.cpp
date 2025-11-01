#include "add.h"
#include "kernels/add.h"

GPUVector<float> vecf_add(const GPUVector<float>& a, const GPUVector<float>& b) {
    const uint64_t n = a.shape(0);
    float *result = gpu_add_f32(a.data(), b.data(), n);
    return make_gpu_vector(result, n);
}

void vecf_add_out(const GPUVector<float>& a, const GPUVector<float>& b, GPUVector<float>& out) {
    // TODO
    const uint64_t n = a.shape(0);
    gpu_add_out_f32(a.data(), b.data(), out.data(), n);
}
