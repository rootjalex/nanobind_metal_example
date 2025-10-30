#include <metal_stdlib>
using namespace metal;

// Simple element-wise vector add: out[i] = inA[i] + inB[i]
kernel void vecf_add(const device float* inA [[ buffer(0) ]],
                     const device float* inB [[ buffer(1) ]],
                     device float*       out [[ buffer(2) ]],
                     constant uint&      N   [[ buffer(3) ]],
                     uint                id  [[ thread_position_in_grid ]]) {
    if (id >= N) return;
    out[id] = inA[id] + inB[id];
}
