#pragma once

#include <cuda_runtime.h>

// Warp-level reductions with shuffles. __shfl_xor_sync lets a thread read a
// register of another thread in the same warp directly, with no shared
// memory and no __syncthreads(). The butterfly pattern (offsets 16, 8, 4, 2, 1)
// takes 5 steps for 32 lanes and leaves the result in EVERY lane.

constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(kFullMask, v, offset);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_xor_sync(kFullMask, v, offset));
    return v;
}
