#pragma once

#include <cuda_runtime.h>

// Row-wise softmax: y[r, c] = exp(x[r, c] - max_r) / sum_c exp(x[r, c] - max_r)
// x and y are row-major [rows x cols], FP32. Subtracting the row max keeps
// exp() from overflowing; -inf inputs (masked positions) produce 0.

enum class SoftmaxVariant : int {
    Naive = 0,       // one thread per row, 3 passes over the row
    Warp = 1,        // one warp per row, 3 passes, shuffle reductions
    WarpOnline = 2,  // one warp per row, online max+sum: 2 passes
};

constexpr int kNumSoftmaxVariants = 3;

const char* softmax_variant_name(SoftmaxVariant v);

void launch_softmax(SoftmaxVariant v, const float* x, float* y, int rows, int cols,
                    cudaStream_t stream = 0);
