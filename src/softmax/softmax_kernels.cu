#include <cmath>

#include "common/warp_reduce.cuh"
#include "softmax/softmax.cuh"

namespace {

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

constexpr int kWarpsPerBlock = 8;  // 256 threads, one row per warp

// ---------------------------------------------------------------------------
// v0: one thread per row
// Simple, but (1) neighbouring threads in a warp read addresses `cols` floats
// apart, so no load is coalesced, and (2) with few rows there are few
// threads: 4096 rows is only 16 blocks of 256 for the whole GPU.
// ---------------------------------------------------------------------------
__global__ void softmax_naive(const float* x, float* y, int rows, int cols) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* in = x + static_cast<size_t>(r) * cols;
    float* out = y + static_cast<size_t>(r) * cols;

    float m = -INFINITY;
    for (int c = 0; c < cols; ++c) m = fmaxf(m, in[c]);
    float s = 0.f;
    for (int c = 0; c < cols; ++c) s += expf(in[c] - m);
    const float inv = 1.f / s;
    for (int c = 0; c < cols; ++c) out[c] = expf(in[c] - m) * inv;
}

// ---------------------------------------------------------------------------
// v1: one warp per row
// Lane i handles columns i, i+32, i+64, ... so on every iteration the 32
// lanes read 32 consecutive floats: one coalesced 128-byte transaction.
// Each lane builds a partial max / sum, then a 5-step shuffle reduction
// combines the 32 partials.
// ---------------------------------------------------------------------------
__global__ void softmax_warp(const float* x, float* y, int rows, int cols) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = blockIdx.x * kWarpsPerBlock + warp;
    if (r >= rows) return;  // whole warp exits together, so shuffles stay safe
    const float* in = x + static_cast<size_t>(r) * cols;
    float* out = y + static_cast<size_t>(r) * cols;

    float m = -INFINITY;
    for (int c = lane; c < cols; c += 32) m = fmaxf(m, in[c]);
    m = warp_reduce_max(m);

    float s = 0.f;
    for (int c = lane; c < cols; c += 32) s += expf(in[c] - m);
    s = warp_reduce_sum(s);

    const float inv = 1.f / s;
    for (int c = lane; c < cols; c += 32) out[c] = expf(in[c] - m) * inv;
}

// ---------------------------------------------------------------------------
// v2: one warp per row, online softmax
// v1 reads the row 3 times (max, sum, write). The online trick computes max
// and sum in ONE pass: keep a running max m and a sum s relative to it, and
// when a bigger value arrives rescale s by exp(m_old - m_new).
// Two partial (m, s) pairs merge the same way, which is what the warp
// reduction does. The same idea is at the heart of FlashAttention.
// ---------------------------------------------------------------------------

// Rescale factor exp(m_old - m_new), defined as 0 when m_old is -inf (empty).
__device__ __forceinline__ float rescale(float m_old, float m_new) {
    return m_old == -INFINITY ? 0.f : expf(m_old - m_new);
}

__global__ void softmax_warp_online(const float* x, float* y, int rows, int cols) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = blockIdx.x * kWarpsPerBlock + warp;
    if (r >= rows) return;
    const float* in = x + static_cast<size_t>(r) * cols;
    float* out = y + static_cast<size_t>(r) * cols;

    float m = -INFINITY, s = 0.f;
    int c = lane;
    // Main loop: 4 independent loads per lane per iteration. A loop where
    // every step depends on the previous (m, s) keeps only one load in
    // flight per thread; issuing 4 at once hides more memory latency.
    for (; c + 96 < cols; c += 128) {
        const float v0 = in[c], v1 = in[c + 32], v2 = in[c + 64], v3 = in[c + 96];
        const float m_new = fmaxf(m, fmaxf(fmaxf(v0, v1), fmaxf(v2, v3)));
        if (m_new == -INFINITY) continue;  // all masked so far: nothing to add
        s = s * rescale(m, m_new) + expf(v0 - m_new) + expf(v1 - m_new) + expf(v2 - m_new) +
            expf(v3 - m_new);
        m = m_new;
    }
    for (; c < cols; c += 32) {  // tail
        const float v = in[c];
        const float m_new = fmaxf(m, v);
        if (m_new == -INFINITY) continue;
        s = s * rescale(m, m_new) + expf(v - m_new);
        m = m_new;
    }

    // Merge (m, s) pairs across the warp.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float m_other = __shfl_xor_sync(kFullMask, m, offset);
        const float s_other = __shfl_xor_sync(kFullMask, s, offset);
        const float m_new = fmaxf(m, m_other);
        s = s * rescale(m, m_new) + s_other * rescale(m_other, m_new);
        m = m_new;
    }

    const float inv = 1.f / s;
    for (int c = lane; c < cols; c += 32) out[c] = expf(in[c] - m) * inv;
}

}  // namespace

const char* softmax_variant_name(SoftmaxVariant v) {
    switch (v) {
        case SoftmaxVariant::Naive: return "naive";
        case SoftmaxVariant::Warp: return "warp";
        case SoftmaxVariant::WarpOnline: return "warp_online";
    }
    return "unknown";
}

void launch_softmax(SoftmaxVariant v, const float* x, float* y, int rows, int cols,
                    cudaStream_t stream) {
    switch (v) {
        case SoftmaxVariant::Naive:
            softmax_naive<<<ceil_div(rows, 256), 256, 0, stream>>>(x, y, rows, cols);
            break;
        case SoftmaxVariant::Warp:
            softmax_warp<<<ceil_div(rows, kWarpsPerBlock), kWarpsPerBlock * 32, 0, stream>>>(x, y, rows, cols);
            break;
        case SoftmaxVariant::WarpOnline:
            softmax_warp_online<<<ceil_div(rows, kWarpsPerBlock), kWarpsPerBlock * 32, 0, stream>>>(x, y, rows, cols);
            break;
    }
}
