// metal_utils.h
#pragma once

#include "metal_impl.h"

#include <iostream>

#define CHECK_METAL_ERROR(err)                               \
    do {                                                     \
        if (err) {                                           \
            std::cerr << "Metal error: "                     \
                      << err->localizedDescription()->utf8String() \
                      << std::endl;                          \
            exit(1);                                         \
        }                                                    \
    } while(0)

NS::SharedPtr<MTL::Device> getDevice();

NS::SharedPtr<MTL::Library> loadKernelLibrary(NS::SharedPtr<MTL::Device> device, const std::string& path);

NS::SharedPtr<MTL::Function> loadFunction(NS::SharedPtr<MTL::Library> library, const std::string& name);

static NS::SharedPtr<MTL::CommandQueue> getCommandQueue() {
    static NS::SharedPtr<MTL::CommandQueue> queue = NS::TransferPtr<MTL::CommandQueue>(getDevice()->newCommandQueue());
    return queue;
}

NS::SharedPtr<MTL::CommandBuffer> getCommandBuffer();

// --- Utility for setting one argument ---
template <typename Arg>
void setBufferOrBytesArgument(MTL::ComputeCommandEncoder* encoder, const Arg& arg, uint32_t index) {
    using BareT = std::remove_cv_t<std::remove_pointer_t<Arg>>;

    if constexpr (std::is_pointer_v<Arg> && std::is_base_of_v<MTL::Buffer, BareT>) {
        // MTL::Buffer* (const or not)
        encoder->setBuffer(const_cast<MTL::Buffer*>(arg), 0, index);
    } else if constexpr (std::is_integral_v<Arg> || std::is_floating_point_v<Arg>) {
        // TODO: for passing structs, may also want: std::is_trivially_copyable_v<BareT> && small size (< 4KB?)
        // Scalar type
        encoder->setBytes(&arg, sizeof(Arg), index);
    } else {
        static_assert(sizeof(Arg) == 0, "Only MTL::Buffer* or scalar types are supported");
    }
}


// --- Variadic helper ---
template <typename... Args>
void setBufferOrBytes(MTL::ComputeCommandEncoder* encoder, const Args&... args) {
    uint32_t index = 0;
    (setBufferOrBytesArgument(encoder, args, index++), ...);
}

// N is the number of threads to launch
template <typename... Args>
void launchFunction(NS::SharedPtr<MTL::Device> device,
                    NS::SharedPtr<MTL::Function> function,
                    const uint32_t n_threads,
                    const Args&... args) {
    auto commandBuffer = getCommandBuffer();
    // Pretty sure this does not need to me memory-managed, tied to commandBuffer lifetime.
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    NS::Error* error = nullptr;
    NS::SharedPtr<MTL::ComputePipelineState> pipelineState =
        NS::TransferPtr<MTL::ComputePipelineState>(device->newComputePipelineState(function.get(), &error));
    CHECK_METAL_ERROR(error);

    encoder->setComputePipelineState(pipelineState.get());

    // Set buffers or bytes for args
    setBufferOrBytes(encoder, args...);

    // Determine threads per threadgroup
    size_t maxThreadsPerThreadgroup = pipelineState->maxTotalThreadsPerThreadgroup();
    size_t threadsPerThreadgroupCount = std::min<size_t>(n_threads, maxThreadsPerThreadgroup);
    MTL::Size threadsPerThreadgroup = MTL::Size(threadsPerThreadgroupCount, 1, 1);

    size_t numThreadgroups = (n_threads + threadsPerThreadgroupCount - 1) / threadsPerThreadgroupCount;
    MTL::Size threadGroupsPerGrid = MTL::Size(numThreadgroups, 1, 1);

    encoder->dispatchThreadgroups(threadGroupsPerGrid, threadsPerThreadgroup);

    encoder->endEncoding();
    commandBuffer->commit();
}

template<typename T>
MTL::Buffer *makeDeviceBuffer(NS::SharedPtr<MTL::Device> device, const size_t &size) {
    return device->newBuffer(NS::UInteger(size * sizeof(T)), MTL::ResourceOptions(MTL::ResourceStorageModeShared));
}

void synchronize();
