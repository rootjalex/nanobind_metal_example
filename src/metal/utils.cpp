// metal_utils.cpp
#include "utils.h"

#include <iostream>
#include <filesystem>
#include <fstream>

NS::SharedPtr<MTL::Device> getDevice() {
    static NS::SharedPtr<MTL::Device> device = NS::TransferPtr<MTL::Device>(MTL::CreateSystemDefaultDevice());
    if (device) {
        return device;
    }
    std::cerr << "Failed to create Metal device" << std::endl;
    exit(-1);
}

NS::SharedPtr<MTL::Library> loadKernelLibrary(NS::SharedPtr<MTL::Device> device, const std::string& relPath) {
    namespace fs = std::filesystem;

    // __FILE__ is the full path to this source file (metal_utils.cpp)
    fs::path currentFile = __FILE__;           
    fs::path baseDir = currentFile.parent_path();

    fs::path fullPath = baseDir / relPath;
    std::ifstream kernelSource(fullPath);
    if (!kernelSource.is_open()) {
        std::cerr << "Failed to open Metal source: " << fullPath << std::endl;
        exit(1);
    }

    std::string source((std::istreambuf_iterator<char>(kernelSource)),
                       std::istreambuf_iterator<char>());

    NS::Error* error = nullptr;
    auto library = NS::TransferPtr(device->newLibrary(
        NS::String::string(source.c_str(), NS::UTF8StringEncoding), 
        nullptr, 
        &error
    ));
    CHECK_METAL_ERROR(error);

    return library;
}

NS::SharedPtr<MTL::Function> loadFunction(NS::SharedPtr<MTL::Library> library, const std::string& name) {
    return NS::TransferPtr<MTL::Function>(
        library->newFunction(NS::String::string(name.c_str(), NS::UTF8StringEncoding)
    ));
}

NS::SharedPtr<MTL::CommandBuffer> getCommandBuffer() {
    return NS::TransferPtr<MTL::CommandBuffer>(getCommandQueue()->commandBuffer());
}

void synchronize() {
    // Get new command buffer and wait on it.
    // All prior command buffers must finish first.
    auto commandBuffer = getCommandBuffer();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}