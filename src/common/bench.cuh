#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "common/cuda_utils.cuh"

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

// Writes over a buffer larger than the L2 cache before each timed run, so
// every kernel starts with a cold cache. Without this, small inputs stay in
// L2 between iterations and the kernel looks faster than it would be in a
// real model, where other work evicts the data.
class L2Flusher {
public:
    L2Flusher() {
        int dev = 0, l2 = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
        bytes_ = static_cast<size_t>(l2) * 2;
        CUDA_CHECK(cudaMalloc(&buf_, bytes_));
    }
    ~L2Flusher() { cudaFree(buf_); }
    void flush() { CUDA_CHECK(cudaMemsetAsync(buf_, 0, bytes_)); }

private:
    void* buf_ = nullptr;
    size_t bytes_ = 0;
};

// Times `launch` with CUDA events and returns the median in milliseconds.
// Events are recorded on the GPU timeline, so they measure kernel time only,
// not launch overhead on the CPU. The median ignores outliers (e.g. the OS
// briefly using the GPU for the display).
template <typename F>
float time_kernel_ms(F launch, int warmup = 3, int iters = 20) {
    static L2Flusher flusher;
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK_LAST_KERNEL();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times(iters);
    for (int i = 0; i < iters; ++i) {
        flusher.flush();
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    CUDA_CHECK_LAST_KERNEL();
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(times.begin(), times.end());
    return times[iters / 2];
}

// ---------------------------------------------------------------------------
// Data and verification
// ---------------------------------------------------------------------------

inline std::vector<float> random_vector(size_t n, unsigned seed, float lo = -1.f, float hi = 1.f) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(gen);
    return v;
}

struct CompareResult {
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;  // relative to the largest |reference| value
    bool passed = false;
};

// Element-wise |out - ref| <= atol + rtol * |ref| (same rule as numpy.allclose).
inline CompareResult compare(const std::vector<float>& out, const std::vector<float>& ref,
                             double atol, double rtol) {
    CompareResult r;
    r.passed = true;
    double max_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs(static_cast<double>(out[i]) - ref[i]);
        max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref[i])));
        r.max_abs_err = std::max(r.max_abs_err, d);
        if (!(d <= atol + rtol * std::fabs(static_cast<double>(ref[i])))) r.passed = false;  // also catches NaN
    }
    r.max_rel_err = max_ref > 0 ? r.max_abs_err / max_ref : r.max_abs_err;
    return r;
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

// Returns the value after `--name`, or nullptr.
inline const char* arg_value(int argc, char** argv, const char* name) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    return nullptr;
}

inline bool has_flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return true;
    return false;
}

// Theoretical DRAM bandwidth in GB/s (2 transfers per clock x bus width).
// Memory-bound kernels (softmax, LayerNorm) are judged against this ceiling.
inline double peak_bandwidth_gbs() {
    int dev = 0, clock_khz = 0, bus_bits = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    return 2.0 * clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
}

inline std::string device_name() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    return p.name;
}
