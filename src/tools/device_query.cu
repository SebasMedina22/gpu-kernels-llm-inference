// Prints the hardware limits that matter when designing kernels:
// how many threads fit in a block / an SM, how much shared memory and how many
// registers are available, and the theoretical memory bandwidth.

#include <cstdio>

#include "common/cuda_utils.cuh"

static int attr(cudaDeviceAttr a, int dev) {
    int v = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&v, a, dev));
    return v;
}

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        std::printf("No CUDA device found.\n");
        return 1;
    }

    int runtime = 0, driver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    std::printf("CUDA runtime %d.%d, driver supports %d.%d\n\n",
                runtime / 1000, (runtime % 1000) / 10,
                driver / 1000, (driver % 1000) / 10);

    for (int dev = 0; dev < count; ++dev) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

        // Clock rates come from the attribute API: the cudaDeviceProp clock
        // fields were removed in CUDA 13.
        const int mem_clock_khz = attr(cudaDevAttrMemoryClockRate, dev);
        const int bus_width_bits = attr(cudaDevAttrGlobalMemoryBusWidth, dev);
        const int sm_clock_khz = attr(cudaDevAttrClockRate, dev);
        // 2x for double data rate. Approximate: some memory types report
        // clocks differently, so treat this as a ballpark ceiling.
        const double peak_bw_gbs =
            2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8.0) / 1e9;

        std::printf("Device %d: %s\n", dev, p.name);
        std::printf("  Compute capability        : %d.%d\n", p.major, p.minor);
        std::printf("  Streaming multiprocessors : %d\n", p.multiProcessorCount);
        std::printf("  SM clock                  : %.0f MHz\n", sm_clock_khz / 1e3);
        std::printf("  Warp size                 : %d\n", p.warpSize);
        std::printf("  Max threads per block     : %d\n", p.maxThreadsPerBlock);
        std::printf("  Max threads per SM        : %d\n", p.maxThreadsPerMultiProcessor);
        std::printf("  Max block dims            : %d x %d x %d\n",
                    p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        std::printf("  Max grid dims             : %d x %d x %d\n",
                    p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
        std::printf("  Registers per block       : %d\n", p.regsPerBlock);
        std::printf("  Registers per SM          : %d\n", p.regsPerMultiprocessor);
        std::printf("  Shared mem per block      : %zu KB (opt-in max %zu KB)\n",
                    p.sharedMemPerBlock / 1024, p.sharedMemPerBlockOptin / 1024);
        std::printf("  Shared mem per SM         : %zu KB\n",
                    p.sharedMemPerMultiprocessor / 1024);
        std::printf("  L2 cache                  : %d KB\n", p.l2CacheSize / 1024);
        std::printf("  Global memory             : %.2f GB\n",
                    p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        std::printf("  Memory bus width          : %d bits\n", bus_width_bits);
        std::printf("  Memory clock              : %.0f MHz\n", mem_clock_khz / 1e3);
        std::printf("  Peak memory bandwidth     : ~%.1f GB/s (theoretical)\n\n",
                    peak_bw_gbs);
    }
    return 0;
}
