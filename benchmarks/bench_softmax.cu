// Softmax: correctness against a CPU reference, then timing of each variant.
//
//   bench_softmax                              verify + benchmark default shapes
//   bench_softmax --rows 4096 --cols 1024,4096 custom shapes
//   bench_softmax --csv out.csv                also write results as CSV
//   bench_softmax --profile --cols 4096        run each kernel once (for Nsight Compute)

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "common/bench.cuh"
#include "softmax/softmax.cuh"

static std::vector<float> cpu_softmax(const std::vector<float>& x, int rows, int cols) {
    std::vector<float> y(x.size());
    for (int r = 0; r < rows; ++r) {
        const float* in = &x[static_cast<size_t>(r) * cols];
        double m = -INFINITY;
        for (int c = 0; c < cols; ++c) m = std::max(m, static_cast<double>(in[c]));
        double s = 0.0;
        for (int c = 0; c < cols; ++c) s += std::exp(in[c] - m);
        for (int c = 0; c < cols; ++c)
            y[static_cast<size_t>(r) * cols + c] = static_cast<float>(std::exp(in[c] - m) / s);
    }
    return y;
}

// Inputs with a wide range and some masked (-inf) entries, like attention
// scores. Every 7th row gets a +1000 offset: a softmax without the
// max-subtraction would overflow exp() there and return inf/NaN.
static std::vector<float> make_input(int rows, int cols, unsigned seed) {
    auto x = random_vector(static_cast<size_t>(rows) * cols, seed, -10.f, 10.f);
    for (int r = 0; r < rows; ++r) {
        float* row = &x[static_cast<size_t>(r) * cols];
        if (r % 7 == 0)
            for (int c = 0; c < cols; ++c) row[c] += 1000.f;
        if (r % 5 == 0)  // causal-style mask on the last half (always keeps column 0)
            for (int c = cols / 2 + 1; c < cols; ++c) row[c] = -INFINITY;
    }
    return x;
}

static std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) v.push_back(std::stoi(item));
    return v;
}

struct DeviceBufs {
    float *x = nullptr, *y = nullptr;
    size_t n = 0;
    explicit DeviceBufs(const std::vector<float>& hx) : n(hx.size()) {
        CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&y, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(x, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~DeviceBufs() { cudaFree(x); cudaFree(y); }
    std::vector<float> run_and_download(SoftmaxVariant v, int rows, int cols) {
        CUDA_CHECK(cudaMemset(y, 0xFF, n * sizeof(float)));  // NaN, so unwritten outputs fail
        launch_softmax(v, x, y, rows, cols);
        CUDA_CHECK_LAST_KERNEL();
        std::vector<float> h(n);
        CUDA_CHECK(cudaMemcpy(h.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost));
        return h;
    }
};

int main(int argc, char** argv) {
    const char* rows_arg = arg_value(argc, argv, "--rows");
    const char* cols_arg = arg_value(argc, argv, "--cols");
    const int rows = rows_arg ? std::atoi(rows_arg) : 4096;
    const std::vector<int> col_list = cols_arg ? parse_list(cols_arg)
                                               : std::vector<int>{128, 512, 1024, 4096, 16384, 32768};

    if (has_flag(argc, argv, "--profile")) {
        DeviceBufs d(make_input(rows, col_list[0], 1));
        for (int v = 0; v < kNumSoftmaxVariants; ++v)
            launch_softmax(static_cast<SoftmaxVariant>(v), d.x, d.y, rows, col_list[0]);
        CUDA_CHECK_LAST_KERNEL();
        std::printf("profile run done (%d x %d)\n", rows, col_list[0]);
        return 0;
    }

    bool all_passed = true;
    const double atol = 1e-6, rtol = 1e-4;

    // ---- 1. Correctness on awkward shapes ----
    std::printf("== Correctness vs CPU reference (double) ==\n");
    const int shapes[][2] = {{1, 1}, {3, 31}, {17, 33}, {100, 1000}, {64, 5000}};
    for (const auto& s : shapes) {
        auto hx = make_input(s[0], s[1], 3);
        auto ref = cpu_softmax(hx, s[0], s[1]);
        DeviceBufs d(hx);
        for (int v = 0; v < kNumSoftmaxVariants; ++v) {
            const auto var = static_cast<SoftmaxVariant>(v);
            auto r = compare(d.run_and_download(var, s[0], s[1]), ref, atol, rtol);
            all_passed &= r.passed;
            std::printf("  %5d x %5d  %-12s max_abs_err=%.2e  %s\n", s[0], s[1],
                        softmax_variant_name(var), r.max_abs_err, r.passed ? "PASS" : "FAIL");
        }
    }

    // ---- 2. Benchmark (verified at every shape before timing) ----
    const char* csv_path = arg_value(argc, argv, "--csv");
    FILE* csv = nullptr;
    if (csv_path) {
        csv = std::fopen(csv_path, "w");
        if (!csv) { std::fprintf(stderr, "cannot open %s\n", csv_path); return 1; }
        std::fprintf(csv, "op,variant,rows,cols,ms,gbps,pct_peak_bw,max_abs_err,gpu\n");
    }
    const std::string gpu = device_name();
    const double peak = peak_bandwidth_gbs();

    std::printf("\n== Benchmark on %s (median of CUDA-event timings, cold L2) ==\n", gpu.c_str());
    std::printf("  Effective bandwidth = (read x once + write y once) / time. Peak ~%.0f GB/s.\n", peak);
    std::printf("  %6s %6s  %-12s %10s %10s %8s\n", "rows", "cols", "variant", "ms", "GB/s", "% peak");
    for (int cols : col_list) {
        auto hx = make_input(rows, cols, 7);
        auto ref = cpu_softmax(hx, rows, cols);
        DeviceBufs d(hx);
        const double bytes = 2.0 * rows * static_cast<double>(cols) * sizeof(float);

        for (int v = 0; v < kNumSoftmaxVariants; ++v) {
            const auto var = static_cast<SoftmaxVariant>(v);
            auto r = compare(d.run_and_download(var, rows, cols), ref, atol, rtol);
            if (!r.passed) {
                all_passed = false;
                std::printf("  %6d %6d  %-12s FAILED verification (max_abs_err=%.2e), not timed\n",
                            rows, cols, softmax_variant_name(var), r.max_abs_err);
                continue;
            }
            auto run = [&] { launch_softmax(var, d.x, d.y, rows, cols); };
            const float probe = time_kernel_ms(run, 1, 1);
            const float ms = time_kernel_ms(run, 2, probe > 50.f ? 5 : 30);
            const double gbps = bytes / (ms * 1e-3) / 1e9;
            std::printf("  %6d %6d  %-12s %10.4f %10.1f %7.1f%%\n", rows, cols,
                        softmax_variant_name(var), ms, gbps, 100.0 * gbps / peak);
            if (csv)
                std::fprintf(csv, "softmax,%s,%d,%d,%.6f,%.3f,%.2f,%.3e,%s\n", softmax_variant_name(var),
                             rows, cols, ms, gbps, 100.0 * gbps / peak, r.max_abs_err, gpu.c_str());
        }
    }
    if (csv) std::fclose(csv);

    std::printf("\n%s\n", all_passed ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return all_passed ? 0 : 1;
}
