#pragma once
#include <cstdint>

float *gpu_add_f32(const float *x, const float *y, const uint64_t n);

void gpu_add_out_f32(const float *x, const float *y, float *result, const uint64_t n);
