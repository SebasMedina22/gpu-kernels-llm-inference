#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap every CUDA runtime call: CUDA_CHECK(cudaMalloc(&p, bytes));
// CUDA calls return error codes instead of throwing, so unchecked errors
// silently corrupt later results.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",              \
                         cudaGetErrorName(err_), __FILE__, __LINE__,          \
                         cudaGetErrorString(err_));                           \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// Kernel launches don't return an error code. Call this right after a launch:
// cudaGetLastError catches bad launch configs, cudaDeviceSynchronize catches
// errors raised while the kernel ran (e.g. out-of-bounds access).
#define CUDA_CHECK_LAST_KERNEL()                                              \
    do {                                                                      \
        CUDA_CHECK(cudaGetLastError());                                       \
        CUDA_CHECK(cudaDeviceSynchronize());                                  \
    } while (0)
