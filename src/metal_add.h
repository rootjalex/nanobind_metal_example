#pragma once

#include "utils.h"

// Allocates and returns a new buffer
GPUVector<float> vecf_add(const GPUVector<float>& a, const GPUVector<float>& b);

// Writes result into a preallocated buffer
void vecf_add_out(const GPUVector<float>& a, const GPUVector<float>& b, GPUVector<float>& out);
