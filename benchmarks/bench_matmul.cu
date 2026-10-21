// Matmul: correctness checks, then timing of each variant against cuBLAS.
//
//   bench_matmul                          verify + benchmark default sizes
//   bench_matmul --sizes 512,1024         custom square sizes
//   bench_matmul --csv out.csv            also write results as CSV
//   bench_matmul --profile --size 2048    run each kernel once (for Nsight Compute)

#include <cublas_v2.h>

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "common/bench.cuh"
#include "matmul/matmul.cuh"

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t s_ = (call);                                            \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_,        \
                         __FILE__, __LINE__);                                  \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// cuBLAS is column-major. A row-major matrix is the transpose of the same
// memory read column-major, so C = A*B (row-major) is C^T = B^T * A^T
// (column-major): pass B first and swap M and N. No data is copied.
static void cublas_matmul(cublasHandle_t h, const float* A, const float* B, float* C,
                          int M, int N, int K) {
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

static std::vector<float> cpu_matmul(const std::vector<float>& A, const std::vector<float>& B,
                                     int M, int N, int K) {
    // Accumulate in double so the reference is more accurate than any FP32 result.
    std::vector<double> acc(static_cast<size_t>(M) * N, 0.0);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            const double a = A[static_cast<size_t>(i) * K + k];
            const float* b = &B[static_cast<size_t>(k) * N];
            double* c = &acc[static_cast<size_t>(i) * N];
            for (int j = 0; j < N; ++j) c[j] += a * b[j];
        }
    std::vector<float> out(acc.size());
    for (size_t i = 0; i < acc.size(); ++i) out[i] = static_cast<float>(acc[i]);
    return out;
}

struct DeviceMats {
    float *A = nullptr, *B = nullptr, *C = nullptr;
    DeviceMats(const std::vector<float>& hA, const std::vector<float>& hB, size_t c_elems) {
        CUDA_CHECK(cudaMalloc(&A, hA.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B, hB.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&C, c_elems * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(A, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~DeviceMats() { cudaFree(A); cudaFree(B); cudaFree(C); }
    std::vector<float> download_c(size_t n) const {
        std::vector<float> h(n);
        CUDA_CHECK(cudaMemcpy(h.data(), C, n * sizeof(float), cudaMemcpyDeviceToHost));
        return h;
    }
    void clear_c(size_t n) { CUDA_CHECK(cudaMemset(C, 0xFF, n * sizeof(float))); }  // NaN pattern
};

// FP32 sums of K products: rounding error grows with K, so the tolerance does too.
static CompareResult check(const std::vector<float>& out, const std::vector<float>& ref, int K) {
    return compare(out, ref, /*atol=*/1e-5 * K, /*rtol=*/1e-4);
}

static std::vector<int> parse_sizes(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) v.push_back(std::stoi(item));
    return v;
}

int main(int argc, char** argv) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // Plain FP32 math: no TF32 tensor cores, so cuBLAS does the same arithmetic as our kernels.
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    if (has_flag(argc, argv, "--profile")) {
        const char* s = arg_value(argc, argv, "--size");
        const int n = s ? std::atoi(s) : 2048;
        auto hA = random_vector(static_cast<size_t>(n) * n, 1);
        auto hB = random_vector(static_cast<size_t>(n) * n, 2);
        DeviceMats d(hA, hB, static_cast<size_t>(n) * n);
        for (int v = 0; v < kNumMatmulVariants; ++v)
            launch_matmul(static_cast<MatmulVariant>(v), d.A, d.B, d.C, n, n, n);
        cublas_matmul(handle, d.A, d.B, d.C, n, n, n);
        CUDA_CHECK_LAST_KERNEL();
        std::printf("profile run done (n=%d)\n", n);
        CUBLAS_CHECK(cublasDestroy(handle));
        return 0;
    }

    bool all_passed = true;

    // ---- 1. Correctness against a CPU reference, including awkward shapes ----
    std::printf("== Correctness vs CPU reference (double accumulation) ==\n");
    const int shapes[][3] = {{128, 128, 128}, {257, 129, 65}, {512, 512, 512}, {1000, 777, 333}, {1, 300, 1}};
    for (const auto& s : shapes) {
        const int M = s[0], N = s[1], K = s[2];
        auto hA = random_vector(static_cast<size_t>(M) * K, 1);
        auto hB = random_vector(static_cast<size_t>(K) * N, 2);
        auto ref = cpu_matmul(hA, hB, M, N, K);
        const size_t nc = static_cast<size_t>(M) * N;
        DeviceMats d(hA, hB, nc);

        for (int v = 0; v <= kNumMatmulVariants; ++v) {
            d.clear_c(nc);
            const char* name;
            if (v < kNumMatmulVariants) {
                launch_matmul(static_cast<MatmulVariant>(v), d.A, d.B, d.C, M, N, K);
                name = matmul_variant_name(static_cast<MatmulVariant>(v));
            } else {
                cublas_matmul(handle, d.A, d.B, d.C, M, N, K);
                name = "cublas";
            }
            CUDA_CHECK_LAST_KERNEL();
            auto r = check(d.download_c(nc), ref, K);
            all_passed &= r.passed;
            std::printf("  %4dx%4dx%4d  %-18s max_abs_err=%.2e  %s\n", M, N, K, name,
                        r.max_abs_err, r.passed ? "PASS" : "FAIL");
        }
    }

    // ---- 2. Benchmark (each variant re-verified against cuBLAS at every size) ----
    const char* sizes_arg = arg_value(argc, argv, "--sizes");
    const std::vector<int> sizes = sizes_arg ? parse_sizes(sizes_arg)
                                             : std::vector<int>{256, 512, 1024, 2048, 4096};
    const char* csv_path = arg_value(argc, argv, "--csv");
    FILE* csv = nullptr;
    if (csv_path) {
        csv = std::fopen(csv_path, "w");
        if (!csv) { std::fprintf(stderr, "cannot open %s\n", csv_path); return 1; }
        std::fprintf(csv, "op,variant,M,N,K,ms,gflops,max_abs_err_vs_cublas,gpu\n");
    }
    const std::string gpu = device_name();

    std::printf("\n== Benchmark on %s (FP32, median of CUDA-event timings, cold L2) ==\n", gpu.c_str());
    std::printf("  %6s  %-18s %10s %10s %10s %12s\n", "N", "variant", "ms", "GFLOP/s", "% cuBLAS", "err vs cuBLAS");
    for (int n : sizes) {
        const int M = n, N = n, K = n;
        const size_t nc = static_cast<size_t>(M) * N;
        auto hA = random_vector(static_cast<size_t>(M) * K, 1);
        auto hB = random_vector(static_cast<size_t>(K) * N, 2);
        DeviceMats d(hA, hB, nc);
        const double flops = 2.0 * M * N * K;

        cublas_matmul(handle, d.A, d.B, d.C, M, N, K);
        CUDA_CHECK_LAST_KERNEL();
        const auto ref = d.download_c(nc);
        // A full CPU matmul is too slow at these sizes, so spot-check the
        // cuBLAS reference itself at random entries before trusting it.
        {
            std::mt19937 gen(42);
            double worst = 0.0;
            bool ok = true;
            for (int s = 0; s < 256; ++s) {
                const int i = static_cast<int>(gen() % M), j = static_cast<int>(gen() % N);
                double dot = 0.0;
                for (int k = 0; k < K; ++k)
                    dot += static_cast<double>(hA[static_cast<size_t>(i) * K + k]) * hB[static_cast<size_t>(k) * N + j];
                const double e = std::fabs(ref[static_cast<size_t>(i) * N + j] - dot);
                worst = std::max(worst, e);
                ok &= e <= 1e-5 * K + 1e-4 * std::fabs(dot);
            }
            all_passed &= ok;
            std::printf("  %6d  cuBLAS reference spot-check vs CPU (256 entries): max_abs_err=%.2e %s\n",
                        n, worst, ok ? "PASS" : "FAIL");
        }
        const float cublas_ms = time_kernel_ms([&] { cublas_matmul(handle, d.A, d.B, d.C, M, N, K); });

        for (int v = 0; v <= kNumMatmulVariants; ++v) {
            const bool is_cublas = v == kNumMatmulVariants;
            const auto var = static_cast<MatmulVariant>(v);
            const char* name = is_cublas ? "cublas" : matmul_variant_name(var);
            auto run = [&] {
                if (is_cublas) cublas_matmul(handle, d.A, d.B, d.C, M, N, K);
                else launch_matmul(var, d.A, d.B, d.C, M, N, K);
            };

            // Verify first. A wrong kernel's timing is meaningless.
            double err = 0.0;
            if (!is_cublas) {
                d.clear_c(nc);
                run();
                CUDA_CHECK_LAST_KERNEL();
                auto r = check(d.download_c(nc), ref, K);
                err = r.max_abs_err;
                if (!r.passed) {
                    all_passed = false;
                    std::printf("  %6d  %-18s FAILED verification (max_abs_err=%.2e), not timed\n", n, name, err);
                    continue;
                }
            }

            // Slow kernels at big sizes take seconds per run, so use fewer iterations.
            const float probe = time_kernel_ms(run, 1, 1);
            const int iters = probe > 200.f ? 5 : (probe > 20.f ? 10 : 30);
            const float ms = is_cublas ? cublas_ms : time_kernel_ms(run, 2, iters);
            const double gflops = flops / (ms * 1e-3) / 1e9;
            std::printf("  %6d  %-18s %10.3f %10.1f %9.1f%% %12.2e\n", n, name, ms, gflops,
                        100.0 * cublas_ms / ms, err);
            if (csv)
                std::fprintf(csv, "matmul,%s,%d,%d,%d,%.6f,%.3f,%.3e,%s\n", name, M, N, K, ms, gflops,
                             err, gpu.c_str());
        }
    }
    if (csv) std::fclose(csv);
    CUBLAS_CHECK(cublasDestroy(handle));

    std::printf("\n%s\n", all_passed ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return all_passed ? 0 : 1;
}
