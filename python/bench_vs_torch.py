"""Time our kernels (through the Python bindings) against the PyTorch equivalents.

Timing uses torch.cuda.Event on the current stream, median of N runs, with
the L2 cache flushed before each run. Both sides are called from Python, so
both include similar per-call overhead (output allocation, launch).

    python python/bench_vs_torch.py --csv out.csv
"""

import argparse
import csv
import statistics

import torch

import llm_kernels as lk

torch.backends.cuda.matmul.allow_tf32 = False  # compare FP32 against FP32

_props = torch.cuda.get_device_properties(0)
_flush = torch.empty(2 * getattr(_props, "L2_cache_size", 32 * 2**20) // 4, device="cuda")


def time_ms(fn, warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        _flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    return statistics.median(times)


def _iters_for(fn) -> int:
    probe = time_ms(fn, warmup=1, iters=1)
    return 5 if probe > 200 else (10 if probe > 20 else 30)


def bench_matmul(sizes):
    rows = []
    for n in sizes:
        a = torch.rand(n, n, device="cuda") * 2 - 1
        b = torch.rand(n, n, device="cuda") * 2 - 1
        ref = a @ b
        cases = {f"ours:{v}": (lambda v=v: lk.matmul(a, b, v)) for v in lk.VARIANTS["matmul"]
                 if v != "naive_uncoalesced"}  # too slow to be interesting here; timed in C++
        cases["torch.matmul"] = lambda: a @ b
        for name, fn in cases.items():
            torch.testing.assert_close(fn(), ref, rtol=1e-4, atol=1e-5 * n)  # verify before timing
            ms = time_ms(fn, iters=_iters_for(fn))
            rows.append({"op": "matmul", "impl": name, "shape": f"{n}x{n}x{n}", "ms": ms,
                         "metric": 2 * n**3 / (ms * 1e-3) / 1e9, "unit": "GFLOP/s"})
    return rows


def bench_softmax(rows_n, cols_list):
    rows = []
    for cols in cols_list:
        x = torch.randn(rows_n, cols, device="cuda") * 10
        ref = torch.softmax(x, dim=-1)
        cases = {f"ours:{v}": (lambda v=v: lk.softmax(x, v)) for v in lk.VARIANTS["softmax"]}
        cases["torch.softmax"] = lambda: torch.softmax(x, dim=-1)
        nbytes = 2 * x.numel() * 4
        for name, fn in cases.items():
            torch.testing.assert_close(fn(), ref, rtol=1e-4, atol=1e-6)
            ms = time_ms(fn, iters=_iters_for(fn))
            rows.append({"op": "softmax", "impl": name, "shape": f"{rows_n}x{cols}", "ms": ms,
                         "metric": nbytes / (ms * 1e-3) / 1e9, "unit": "GB/s"})
    return rows


def bench_layernorm(rows_n, hidden_list):
    rows = []
    for h in hidden_list:
        x = torch.randn(rows_n, h, device="cuda") + 5
        w = torch.rand(h, device="cuda") + 0.5
        b = torch.randn(h, device="cuda")
        ref = torch.nn.functional.layer_norm(x, (h,), w, b, eps=1e-5)
        cases = {f"ours:{v}": (lambda v=v: lk.layer_norm(x, w, b, 1e-5, v)) for v in lk.VARIANTS["layernorm"]}
        cases["torch.layer_norm"] = lambda: torch.nn.functional.layer_norm(x, (h,), w, b, eps=1e-5)
        nbytes = (2 * x.numel() + 2 * h) * 4
        for name, fn in cases.items():
            torch.testing.assert_close(fn(), ref, rtol=1e-4, atol=1e-4)
            ms = time_ms(fn, iters=_iters_for(fn))
            rows.append({"op": "layernorm", "impl": name, "shape": f"{rows_n}x{h}", "ms": ms,
                         "metric": nbytes / (ms * 1e-3) / 1e9, "unit": "GB/s"})
    return rows


def run(matmul_sizes=(512, 1024, 2048, 4096), softmax_cols=(128, 1024, 4096, 32768),
        ln_hidden=(768, 2048, 4096, 8192), rows_n=4096, csv_path=None):
    results = bench_matmul(matmul_sizes) + bench_softmax(rows_n, softmax_cols) + bench_layernorm(rows_n, ln_hidden)
    gpu = torch.cuda.get_device_name(0)
    for r in results:
        r["gpu"] = gpu
        r["torch"] = torch.__version__
    print(f"{'op':<10} {'impl':<24} {'shape':<16} {'ms':>10} {'metric':>10}")
    for r in results:
        print(f"{r['op']:<10} {r['impl']:<24} {r['shape']:<16} {r['ms']:>10.4f} {r['metric']:>10.1f} {r['unit']}")
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv")
    args = p.parse_args()
    run(csv_path=args.csv)
