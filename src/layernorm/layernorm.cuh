#pragma once

#include <cuda_runtime.h>

// LayerNorm over the last dimension:
//   y[r, :] = (x[r, :] - mean_r) / sqrt(var_r + eps) * gamma + beta
// x and y are row-major [rows x hidden], gamma and beta are [hidden], FP32.
// var is the biased variance (divide by hidden), matching torch.nn.LayerNorm.

enum class LayerNormVariant : int {
    Naive = 0,      // one thread per row
    Warp = 1,       // one warp per row, shuffle reductions, re-reads x from global
    BlockSmem = 2,  // one block per row, row cached in shared memory (x read once)
};

constexpr int kNumLayerNormVariants = 3;

const char* layernorm_variant_name(LayerNormVariant v);

// Returns cudaErrorInvalidValue if `hidden` is too large for BlockSmem's
// shared-memory row cache.
cudaError_t launch_layernorm(LayerNormVariant v, const float* x, const float* gamma,
                             const float* beta, float* y, int rows, int hidden, float eps,
                             cudaStream_t stream = 0);
