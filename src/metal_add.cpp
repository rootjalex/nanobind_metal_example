#include "metal_add.h"

#include "metal_utils.h"

NS::SharedPtr<MTL::Library> getMetalLibrary() {
    // Must be static for efficiency
    static NS::SharedPtr<MTL::Library> library = loadKernelLibrary(getDevice(), "kernels/add.metal");
    return library;
}

NS::SharedPtr<MTL::Function> getVecAddFunction() {
    // Must be static for efficiency
    static NS::SharedPtr<MTL::Function> func = loadFunction(getMetalLibrary(), "vecf_add");
    return func;
}

void launch_add(NS::SharedPtr<MTL::Device> device, const MTL::Buffer *a, const MTL::Buffer *b, const MTL::Buffer *out, const size_t N) {
    auto function = getVecAddFunction();
    uint nValue = static_cast<uint>(N);
    const uint32_t N_value = static_cast<uint32_t>(N);
    launchFunction(device, function, N, a, b, out, N_value);
}

// Allocates and returns a new buffer
GPUVector<float> vecf_add(const GPUVector<float>& a, const GPUVector<float>& b) {
    auto device = getDevice();
    MTL::Buffer *output = makeDeviceBuffer<float>(device, a.shape(0));
    launch_add(device, get_mtl_buffer(a), get_mtl_buffer(b), output, a.shape(0));
    return make_gpu_vector((float*)output, a.shape(0));
}

// Writes result into a preallocated buffer
void vecf_add_out(const GPUVector<float>& a, const GPUVector<float>& b, GPUVector<float>& out) {
    auto device = getDevice();
    launch_add(device, get_mtl_buffer(a), get_mtl_buffer(b), get_mtl_buffer(out), a.shape(0));
}
