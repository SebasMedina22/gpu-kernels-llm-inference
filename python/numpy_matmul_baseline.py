"""CPU baseline: FP32 matmul with NumPy (multi-threaded BLAS on the CPU).

Usage:
    python python/numpy_matmul_baseline.py                 # default sizes
    python python/numpy_matmul_baseline.py --sizes 512,1024 --csv out.csv
"""

import argparse
import csv
import platform
import time

import numpy as np


def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import subprocess

            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor() or platform.machine()


def time_matmul(n: int, repeats: int) -> float:
    """Median wall-clock time in ms of an n x n x n FP32 matmul."""
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, (n, n)).astype(np.float32)
    b = rng.uniform(-1, 1, (n, n)).astype(np.float32)
    a @ b  # warm-up (thread pool start, page faults)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        a @ b
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def run(sizes, csv_path=None):
    cpu = cpu_name()
    rows = []
    print(f"NumPy {np.__version__} on {cpu}")
    print(f"{'N':>6} {'ms':>10} {'GFLOP/s':>10}")
    for n in sizes:
        repeats = 3 if n >= 4096 else 10
        ms = time_matmul(n, repeats)
        gflops = 2 * n**3 / (ms * 1e-3) / 1e9
        print(f"{n:>6} {ms:>10.3f} {gflops:>10.1f}")
        rows.append({"op": "matmul", "variant": "numpy_cpu", "M": n, "N": n, "K": n,
                     "ms": ms, "gflops": gflops, "device": cpu})
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="256,512,1024,2048,4096")
    p.add_argument("--csv")
    args = p.parse_args()
    run([int(s) for s in args.sizes.split(",")], args.csv)
