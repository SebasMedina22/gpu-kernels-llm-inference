#include "common/warp_reduce.cuh"
#include "layernorm/layernorm.cuh"

namespace {

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

constexpr int kWarpsPerBlock = 8;
constexpr int kBlockThreads = 256;

// ---------------------------------------------------------------------------
// v0: one thread per row (uncoalesced, little parallelism; see softmax_naive)
// Two-pass statistics: mean first, then the mean of squared deviations.
// The one-pass E[x^2] - E[x]^2 formula loses precision when |mean| >> std.
// ---------------------------------------------------------------------------
__global__ void layernorm_naive(const float* x, const float* gamma, const float* beta, float* y,
                                int rows, int hidden, float eps) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* in = x + static_cast<size_t>(r) * hidden;
    float* out = y + static_cast<size_t>(r) * hidden;

    float sum = 0.f;
    for (int c = 0; c < hidden; ++c) sum += in[c];
    const float mean = sum / hidden;
    float sq = 0.f;
    for (int c = 0; c < hidden; ++c) {
        const float d = in[c] - mean;
        sq += d * d;
    }
    const float rstd = rsqrtf(sq / hidden + eps);
    for (int c = 0; c < hidden; ++c) out[c] = (in[c] - mean) * rstd * gamma[c] + beta[c];
}

// ---------------------------------------------------------------------------
// v1: one warp per row
// Coalesced strided access (lane i reads i, i+32, ...), 4 independent loads
// per iteration in the reduction loops (lesson from softmax_warp_online),
// shuffle reductions. Reads x three times; the 2nd and 3rd reads hit cache
// only if the row is still there.
// ---------------------------------------------------------------------------
__global__ void layernorm_warp(const float* x, const float* gamma, const float* beta, float* y,
                               int rows, int hidden, float eps) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = blockIdx.x * kWarpsPerBlock + warp;
    if (r >= rows) return;
    const float* in = x + static_cast<size_t>(r) * hidden;
    float* out = y + static_cast<size_t>(r) * hidden;

    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;  // separate accumulators: no dependency chain
    int c = lane;
    for (; c + 96 < hidden; c += 128) {
        s0 += in[c];
        s1 += in[c + 32];
        s2 += in[c + 64];
        s3 += in[c + 96];
    }
    for (; c < hidden; c += 32) s0 += in[c];
    const float mean = warp_reduce_sum((s0 + s1) + (s2 + s3)) / hidden;

    s0 = s1 = s2 = s3 = 0.f;
    c = lane;
    for (; c + 96 < hidden; c += 128) {
        const float d0 = in[c] - mean, d1 = in[c + 32] - mean;
        const float d2 = in[c + 64] - mean, d3 = in[c + 96] - mean;
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    for (; c < hidden; c += 32) {
        const float d = in[c] - mean;
        s0 += d * d;
    }
    const float rstd = rsqrtf(warp_reduce_sum((s0 + s1) + (s2 + s3)) / hidden + eps);

    for (c = lane; c < hidden; c += 32) out[c] = (in[c] - mean) * rstd * gamma[c] + beta[c];
}

// ---------------------------------------------------------------------------
// v2: one block per row, row cached in shared memory
// The 256 threads copy the row into shared memory once (coalesced), then
// both reductions and the output pass read shared memory instead of global.
// DRAM traffic is the minimum: read x once, write y once.
// Block-wide reduction = warp shuffle, then warp 0 reduces the 8 partials.
// ---------------------------------------------------------------------------
__device__ float block_reduce_sum(float v, float* scratch /* [32] */) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    v = warp_reduce_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    const int num_warps = blockDim.x / 32;
    v = (lane < num_warps) ? scratch[lane] : 0.f;
    if (warp == 0) v = warp_reduce_sum(v);
    if (threadIdx.x == 0) scratch[0] = v;
    __syncthreads();
    const float total = scratch[0];
    __syncthreads();  // scratch can be reused by the next call
    return total;
}

__global__ void layernorm_block_smem(const float* x, const float* gamma, const float* beta,
                                     float* y, int rows, int hidden, float eps) {
    extern __shared__ float row[];  // [hidden], size set at launch
    __shared__ float scratch[32];
    const int r = blockIdx.x;
    const float* in = x + static_cast<size_t>(r) * hidden;
    float* out = y + static_cast<size_t>(r) * hidden;

    float sum = 0.f;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        const float v = in[c];
        row[c] = v;
        sum += v;
    }
    const float mean = block_reduce_sum(sum, scratch) / hidden;  // its __syncthreads also publishes row[]

    float sq = 0.f;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        const float d = row[c] - mean;
        sq += d * d;
    }
    const float rstd = rsqrtf(block_reduce_sum(sq, scratch) / hidden + eps);

    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
        out[c] = (row[c] - mean) * rstd * gamma[c] + beta[c];
}

}  // namespace

const char* layernorm_variant_name(LayerNormVariant v) {
    switch (v) {
        case LayerNormVariant::Naive: return "naive";
        case LayerNormVariant::Warp: return "warp";
        case LayerNormVariant::BlockSmem: return "block_smem";
    }
    return "unknown";
}

cudaError_t launch_layernorm(LayerNormVariant v, const float* x, const float* gamma,
                             const float* beta, float* y, int rows, int hidden, float eps,
                             cudaStream_t stream) {
    switch (v) {
        case LayerNormVariant::Naive:
            layernorm_naive<<<ceil_div(rows, 256), 256, 0, stream>>>(x, gamma, beta, y, rows, hidden, eps);
            break;
        case LayerNormVariant::Warp:
            layernorm_warp<<<ceil_div(rows, kWarpsPerBlock), kWarpsPerBlock * 32, 0, stream>>>(
                x, gamma, beta, y, rows, hidden, eps);
            break;
        case LayerNormVariant::BlockSmem: {
            const size_t smem = static_cast<size_t>(hidden) * sizeof(float);
            if (smem > 48 * 1024) {
                // Above 48 KB of dynamic shared memory a kernel must opt in.
                int dev = 0, max_optin = 0;
                cudaGetDevice(&dev);
                cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
                if (smem + 32 * sizeof(float) > static_cast<size_t>(max_optin)) return cudaErrorInvalidValue;
                cudaError_t e = cudaFuncSetAttribute(layernorm_block_smem,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     static_cast<int>(smem));
                if (e != cudaSuccess) return e;
            }
            layernorm_block_smem<<<rows, kBlockThreads, smem, stream>>>(x, gamma, beta, y, rows, hidden, eps);
            break;
        }
    }
    return cudaGetLastError();
}
