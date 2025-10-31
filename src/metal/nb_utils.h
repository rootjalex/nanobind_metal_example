#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "metal_impl.h"

#include <iostream>

namespace nb = nanobind;

template<typename T>
using GPUVector = nb::ndarray<nb::pytorch, T, nb::shape<-1>, nb::c_contig, nb::device::metal>;

template<typename T>
using CPUVector = nb::ndarray<nb::pytorch, T, nb::shape<-1>, nb::c_contig, nb::device::cpu>;

static void releaseBuffer(void* ptr) noexcept {
    auto b = static_cast<MTL::Buffer*>(const_cast<void*>(ptr));
    if (b) {
        b->release();  // release the Metal buffer
    }
}

template<typename T>
GPUVector<T> make_gpu_vector(T* ptr, const size_t n) {
    nb::capsule owner(ptr, releaseBuffer);

    const size_t shape[1] = {n};

    return GPUVector<T>(ptr, /* ndim = */ 1, shape, owner, /* strides */ nullptr, nb::dtype<T>(), /* explicitly set device type */ nb::device::metal::value);
}

template<typename T>
MTL::Buffer* get_mtl_buffer(const GPUVector<T>& gpu_vec) {
    auto ptr = reinterpret_cast<MTL::Buffer*>(gpu_vec.data());
    if (ptr) {
        return ptr;
    }
    std::cerr << "Failed to get mtl buffer from ndarray" << std::endl;
    exit(-1);
}
