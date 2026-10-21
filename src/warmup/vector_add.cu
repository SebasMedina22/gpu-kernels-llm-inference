// Warm-up: c[i] = a[i] + b[i].
// Exercises the full CUDA workflow used by every kernel in this repo:
// allocate on the GPU, copy inputs over, launch, copy back, verify on the CPU.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common/cuda_utils.cuh"

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // Global index: which block we're in, times threads per block,
    // plus our position inside the block.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // The grid is rounded up to whole blocks, so the last block can have
    // threads past the end of the arrays. They must do nothing.
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    const int n = (argc > 1) ? std::atoi(argv[1]) : 1'000'000;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = std::sin(static_cast<float>(i));
        h_b[i] = std::cos(static_cast<float>(i));
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;  // ceil(n / threads)
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_LAST_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // A single float add is exact on both CPU and GPU, so results must match bit for bit.
    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            if (mismatches < 5) {
                std::printf("Mismatch at %d: gpu=%f cpu=%f\n", i, h_c[i], h_a[i] + h_b[i]);
            }
            ++mismatches;
        }
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    std::printf("vector_add n=%d, %d blocks x %d threads: %s\n", n, blocks, threads,
                mismatches == 0 ? "PASSED" : "FAILED");
    return mismatches == 0 ? 0 : 1;
}
