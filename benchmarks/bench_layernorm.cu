// LayerNorm: correctness against a CPU reference, then timing of each variant.
//
//   bench_layernorm                                  verify + benchmark default shapes
//   bench_layernorm --rows 4096 --hidden 768,4096    custom shapes
//   bench_layernorm --csv out.csv                    also write results as CSV
//   bench_layernorm --profile --hidden 4096          run each kernel once (for Nsight Compute)

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "common/bench.cuh"
#include "layernorm/layernorm.cuh"

constexpr float kEps = 1e-5f;

static std::vector<float> cpu_layernorm(const std::vector<float>& x, const std::vector<float>& g,
                                        const std::vector<float>& b, int rows, int hidden) {
    std::vector<float> y(x.size());
    for (int r = 0; r < rows; ++r) {
        const float* in = &x[static_cast<size_t>(r) * hidden];
        double mean = 0.0;
        for (int c = 0; c < hidden; ++c) mean += in[c];
        mean /= hidden;
        double var = 0.0;
        for (int c = 0; c < hidden; ++c) var += (in[c] - mean) * (in[c] - mean);
        var /= hidden;
        const double rstd = 1.0 / std::sqrt(var + kEps);
        for (int c = 0; c < hidden; ++c)
            y[static_cast<size_t>(r) * hidden + c] = static_cast<float>((in[c] - mean) * rstd * g[c] + b[c]);
    }
    return y;
}

// Activations with a non-zero mean (around 5): a one-pass E[x^2]-E[x]^2
// variance would lose precision here.
static std::vector<float> make_input(int rows, int hidden, unsigned seed) {
    return random_vector(static_cast<size_t>(rows) * hidden, seed, 3.f, 7.f);
}

static std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) v.push_back(std::stoi(item));
    return v;
}

struct DeviceBufs {
    float *x = nullptr, *y = nullptr, *g = nullptr, *b = nullptr;
    size_t n = 0;
    DeviceBufs(const std::vector<float>& hx, const std::vector<float>& hg, const std::vector<float>& hb)
        : n(hx.size()) {
        CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&y, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g, hg.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b, hb.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(x, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g, hg.data(), hg.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b, hb.data(), hb.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~DeviceBufs() { cudaFree(x); cudaFree(y); cudaFree(g); cudaFree(b); }
    void launch(LayerNormVariant v, int rows, int hidden) {
        CUDA_CHECK(launch_layernorm(v, x, g, b, y, rows, hidden, kEps));
    }
    std::vector<float> run_and_download(LayerNormVariant v, int rows, int hidden) {
        CUDA_CHECK(cudaMemset(y, 0xFF, n * sizeof(float)));  // NaN, so unwritten outputs fail
        launch(v, rows, hidden);
        CUDA_CHECK_LAST_KERNEL();
        std::vector<float> h(n);
        CUDA_CHECK(cudaMemcpy(h.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost));
        return h;
    }
};

int main(int argc, char** argv) {
    const char* rows_arg = arg_value(argc, argv, "--rows");
    const char* hidden_arg = arg_value(argc, argv, "--hidden");
    const int rows = rows_arg ? std::atoi(rows_arg) : 4096;
    const std::vector<int> hidden_list = hidden_arg ? parse_list(hidden_arg)
                                                    : std::vector<int>{768, 1024, 2048, 4096, 8192};

    if (has_flag(argc, argv, "--profile")) {
        const int h = hidden_list[0];
        DeviceBufs d(make_input(rows, h, 1), random_vector(h, 2, 0.5f, 1.5f), random_vector(h, 3));
        for (int v = 0; v < kNumLayerNormVariants; ++v) d.launch(static_cast<LayerNormVariant>(v), rows, h);
        CUDA_CHECK_LAST_KERNEL();
        std::printf("profile run done (%d x %d)\n", rows, h);
        return 0;
    }

    bool all_passed = true;
    const double atol = 1e-4, rtol = 1e-4;

    // ---- 1. Correctness on awkward shapes ----
    std::printf("== Correctness vs CPU reference (double) ==\n");
    const int shapes[][2] = {{1, 1}, {3, 31}, {17, 33}, {100, 1000}, {64, 5000}, {8, 20000}};
    for (const auto& s : shapes) {
        auto hx = make_input(s[0], s[1], 4);
        auto hg = random_vector(s[1], 5, 0.5f, 1.5f);
        auto hb = random_vector(s[1], 6);
        auto ref = cpu_layernorm(hx, hg, hb, s[0], s[1]);
        DeviceBufs d(hx, hg, hb);
        for (int v = 0; v < kNumLayerNormVariants; ++v) {
            const auto var = static_cast<LayerNormVariant>(v);
            auto r = compare(d.run_and_download(var, s[0], s[1]), ref, atol, rtol);
            all_passed &= r.passed;
            std::printf("  %5d x %5d  %-11s max_abs_err=%.2e  %s\n", s[0], s[1],
                        layernorm_variant_name(var), r.max_abs_err, r.passed ? "PASS" : "FAIL");
        }
    }

    // ---- 2. Benchmark (verified at every shape before timing) ----
    const char* csv_path = arg_value(argc, argv, "--csv");
    FILE* csv = nullptr;
    if (csv_path) {
        csv = std::fopen(csv_path, "w");
        if (!csv) { std::fprintf(stderr, "cannot open %s\n", csv_path); return 1; }
        std::fprintf(csv, "op,variant,rows,hidden,ms,gbps,pct_peak_bw,max_abs_err,gpu\n");
    }
    const std::string gpu = device_name();
    const double peak = peak_bandwidth_gbs();

    std::printf("\n== Benchmark on %s (median of CUDA-event timings, cold L2) ==\n", gpu.c_str());
    std::printf("  Effective bandwidth = (read x + write y + read gamma, beta) / time. Peak ~%.0f GB/s.\n", peak);
    std::printf("  %6s %6s  %-11s %10s %10s %8s\n", "rows", "hidden", "variant", "ms", "GB/s", "% peak");
    for (int hidden : hidden_list) {
        auto hx = make_input(rows, hidden, 7);
        auto hg = random_vector(hidden, 8, 0.5f, 1.5f);
        auto hb = random_vector(hidden, 9);
        auto ref = cpu_layernorm(hx, hg, hb, rows, hidden);
        DeviceBufs d(hx, hg, hb);
        const double bytes = (2.0 * rows + 2.0) * hidden * sizeof(float);

        for (int v = 0; v < kNumLayerNormVariants; ++v) {
            const auto var = static_cast<LayerNormVariant>(v);
            auto r = compare(d.run_and_download(var, rows, hidden), ref, atol, rtol);
            if (!r.passed) {
                all_passed = false;
                std::printf("  %6d %6d  %-11s FAILED verification (max_abs_err=%.2e), not timed\n",
                            rows, hidden, layernorm_variant_name(var), r.max_abs_err);
                continue;
            }
            auto run = [&] { d.launch(var, rows, hidden); };
            const float probe = time_kernel_ms(run, 1, 1);
            const float ms = time_kernel_ms(run, 2, probe > 50.f ? 5 : 30);
            const double gbps = bytes / (ms * 1e-3) / 1e9;
            std::printf("  %6d %6d  %-11s %10.4f %10.1f %7.1f%%\n", rows, hidden,
                        layernorm_variant_name(var), ms, gbps, 100.0 * gbps / peak);
            if (csv)
                std::fprintf(csv, "layernorm,%s,%d,%d,%.6f,%.3f,%.2f,%.3e,%s\n", layernorm_variant_name(var),
                             rows, hidden, ms, gbps, 100.0 * gbps / peak, r.max_abs_err, gpu.c_str());
        }
    }
    if (csv) std::fclose(csv);

    std::printf("\n%s\n", all_passed ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return all_passed ? 0 : 1;
}
