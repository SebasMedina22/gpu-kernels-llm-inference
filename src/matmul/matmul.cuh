#pragma once

#include <cuda_runtime.h>

// C[M x N] = A[M x K] * B[K x N], FP32, all matrices row-major and contiguous.
// Each launcher enqueues one kernel on `stream` and returns without syncing.

enum class MatmulVariant : int {
    NaiveUncoalesced = 0,  // one thread per output, threadIdx.x walks down rows
    Naive = 1,             // one thread per output, threadIdx.x walks along a row
    Tiled = 2,             // 32x32 shared-memory tiles
    RegisterTiled = 3,     // 128x128 block tile, each thread computes 8x8 outputs
};

constexpr int kNumMatmulVariants = 4;

const char* matmul_variant_name(MatmulVariant v);

void launch_matmul(MatmulVariant v, const float* A, const float* B, float* C,
                   int M, int N, int K, cudaStream_t stream = 0);
