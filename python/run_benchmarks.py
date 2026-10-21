"""Run every benchmark and write results tables + plots to docs/.

    python python/run_benchmarks.py              # full run (needs a Release build)
    python python/run_benchmarks.py --plots-only # regenerate plots/tables from docs/results/*.csv

Outputs:
    docs/results/*.csv      raw results of each benchmark
    docs/results/env.json   GPU, driver, power state, CPU, library versions, date
    docs/plots/*.png        charts
    docs/RESULTS.md         tables generated from the CSVs
"""

import argparse
import csv
import datetime
import json
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "results"
PLOTS = ROOT / "docs" / "plots"
EXE_SUFFIX = ".exe" if platform.system() == "Windows" else ""

sys.path.insert(0, str(ROOT / "python"))


def find_exe(name: str) -> pathlib.Path:
    for d in [ROOT / "build" / "Release", ROOT / "build"]:
        if (d / (name + EXE_SUFFIX)).exists():
            return d / (name + EXE_SUFFIX)
    sys.exit(f"{name} not found: build first with 'cmake --build build --config Release'")


def on_battery() -> bool:
    if platform.system() != "Windows":
        return False
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Battery).BatteryStatus"],
                         capture_output=True, text=True).stdout.strip()
    return out == "1"  # 1 = discharging


def environment() -> dict:
    # enforced.power.limit is the limit actually applied; laptops report power.limit as N/A.
    q = "name,driver_version,enforced.power.limit,power.default_limit,power.max_limit,clocks.max.sm,memory.total"
    smi = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().split(", ")
    import numpy
    import torch
    from numpy_matmul_baseline import cpu_name
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "gpu": smi[0], "driver": smi[1], "gpu_power_limit": smi[2], "gpu_power_default_limit": smi[3],
        "gpu_power_max_limit": smi[4], "gpu_max_sm_clock": smi[5], "gpu_memory": smi[6],
        "power_source": "battery" if on_battery() else "AC",
        "cuda_toolkit": next((l.strip() for l in nvcc.splitlines() if "release" in l), "unknown"),
        "cpu": cpu_name(), "os": platform.platform(),
        "python": platform.python_version(), "numpy": numpy.__version__, "torch": torch.__version__,
    }


# Each result file, and the columns that identify one measurement in it.
KEYS = {
    "matmul_cuda.csv": ("variant", "N"),
    "softmax_cuda.csv": ("variant", "cols"),
    "layernorm_cuda.csv": ("variant", "hidden"),
    "matmul_numpy.csv": ("variant", "N"),
    "vs_torch.csv": ("op", "impl", "shape"),
}


def run_round(raw: pathlib.Path, i: int):
    for name in ["bench_matmul", "bench_softmax", "bench_layernorm"]:
        print(f"\n=== round {i}: {name} ===", flush=True)
        out = raw / f"{name.removeprefix('bench_')}_cuda_r{i}.csv"
        if subprocess.run([str(find_exe(name)), "--csv", str(out)]).returncode != 0:
            sys.exit(f"{name} failed verification; not writing results")

    print(f"\n=== round {i}: NumPy CPU baseline ===", flush=True)
    import numpy_matmul_baseline
    numpy_matmul_baseline.run([256, 512, 1024, 2048, 4096], raw / f"matmul_numpy_r{i}.csv")

    print(f"\n=== round {i}: ours vs PyTorch (Python harness) ===", flush=True)
    import bench_vs_torch
    bench_vs_torch.run(matmul_sizes=(256, 512, 1024, 2048, 4096),
                       softmax_cols=(128, 512, 1024, 4096, 16384, 32768),
                       ln_hidden=(768, 1024, 2048, 4096, 8192),
                       csv_path=raw / f"vs_torch_r{i}.csv")


def merge_rounds(raw: pathlib.Path, rounds: int) -> dict:
    """Keep, per measurement, the round with the lowest time.

    Other work on the machine (browser, display, other GPU apps) can only make
    a kernel slower, never faster, so the fastest of several rounds is the best
    estimate of the kernel's own time. Returns the spread between rounds."""
    spreads = []
    for name, key in KEYS.items():
        stem = name.removesuffix(".csv")
        best, order = {}, []
        all_ms = {}
        for i in range(1, rounds + 1):
            with open(raw / f"{stem}_r{i}.csv", newline="") as f:
                for row in csv.DictReader(f):
                    k = tuple(row[c] for c in key)
                    if k not in best:
                        order.append(k)
                    all_ms.setdefault(k, []).append(float(row["ms"]))
                    if k not in best or float(row["ms"]) < float(best[k]["ms"]):
                        best[k] = row
        spreads += [max(v) / min(v) - 1 for v in all_ms.values()]
        with open(RESULTS / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=best[order[0]].keys())
            w.writeheader()
            w.writerows(best[k] for k in order)
    spreads.sort()
    return {"rounds": rounds, "median_spread_pct": round(100 * spreads[len(spreads) // 2], 1),
            "p90_spread_pct": round(100 * spreads[int(len(spreads) * 0.9)], 1),
            "max_spread_pct": round(100 * spreads[-1], 1)}


TELEMETRY_QUERY = "timestamp,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,utilization.gpu"


def _num(s: str) -> float:
    return float(s.strip().split()[0])


def summarize_telemetry(path: pathlib.Path) -> dict:
    sm, temp, limits = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.split(",")
            try:
                temp_c, clock, limit, util = _num(parts[1]), _num(parts[2]), _num(parts[4]), _num(parts[5])
            except (IndexError, ValueError):
                continue
            limits.append(limit)
            if util >= 50:  # only samples where the GPU was actually busy
                sm.append(clock)
                temp.append(temp_c)
    if not sm:
        return {}
    sm.sort()
    return {"sm_clock_mhz_min": sm[0], "sm_clock_mhz_median": sm[len(sm) // 2], "sm_clock_mhz_max": sm[-1],
            "temp_c_max": max(temp), "busy_samples": len(sm),
            "power_limit_w_min": min(limits), "power_limit_w_max": max(limits)}


def run_all(rounds: int):
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = RESULTS / "raw"
    raw.mkdir(exist_ok=True)
    env = environment()
    print(json.dumps(env, indent=2))

    telemetry = raw / "gpu_telemetry.csv"
    with open(telemetry, "w") as tf:
        logger = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={TELEMETRY_QUERY}", "--format=csv,noheader", "-lms", "500"], stdout=tf)
        try:
            for i in range(1, rounds + 1):
                run_round(raw, i)
        finally:
            logger.terminate()
            logger.wait()

    env["variation"] = merge_rounds(raw, rounds)
    env["telemetry"] = summarize_telemetry(telemetry)
    # Laptop firmware can lower the GPU power limit mid-run (e.g. on a weak
    # charger), which silently slows everything after that point.
    default_w = _num(env["gpu_power_default_limit"])
    env["power_limit_ok"] = env["telemetry"].get("power_limit_w_min", 0) >= default_w
    print(json.dumps({k: env[k] for k in ("variation", "telemetry", "power_limit_ok")}, indent=2))
    (RESULTS / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    if not env["power_limit_ok"]:
        print(f"\nWARNING: GPU power limit fell to {env['telemetry'].get('power_limit_w_min')} W "
              f"(default {default_w} W) during the run. Results are marked invalid in RESULTS.md.")


# ---------------------------------------------------------------------------
# Plots and tables
# ---------------------------------------------------------------------------

def read_csv(name: str) -> list[dict]:
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


# Reference categorical palette (light), fixed order: each implementation keeps its color.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)


def _lines(ax, series: dict, xs_label=str):
    """series: {label: [(x, y), ...]} in fixed order. Direct label at each line's end."""
    for i, (label, pts) in enumerate(series.items()):
        pts = sorted(pts)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=COLORS[i % len(COLORS)], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)
    ax.set_xscale("log", base=2)
    xticks = sorted({p[0] for pts in series.values() for p in pts})
    ax.set_xticks(xticks)
    ax.set_xticklabels([xs_label(x) for x in xticks])
    ax.minorticks_off()


def make_plots(env: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    gpu = env["gpu"]

    # Matmul: C++ harness (includes cuBLAS and the uncoalesced kernel) + NumPy CPU.
    mm = read_csv("matmul_cuda.csv")
    order = ["naive_uncoalesced", "naive", "tiled_smem", "register_tiled", "cublas"]
    series = {v: [(int(r["N"]), float(r["gflops"])) for r in mm if r["variant"] == v] for v in order}
    series["numpy (CPU)"] = [(int(r["N"]), float(r["gflops"])) for r in read_csv("matmul_numpy.csv")]
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    _lines(ax, series)
    ax.set_yscale("log")
    _style(ax, f"FP32 matmul throughput, {gpu}", "N (square N x N x N)", "GFLOP/s (log scale)")
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.12, 1))
    fig.tight_layout()
    fig.savefig(PLOTS / "matmul.png", dpi=150)
    plt.close(fig)

    # Softmax and LayerNorm: Python harness, so ours and PyTorch are measured the same way.
    vt = read_csv("vs_torch.csv")
    peak = _peak_bw()
    for op, torch_name, xlabel in [("softmax", "torch.softmax", "columns per row (4096 rows)"),
                                   ("layernorm", "torch.layer_norm", "hidden size (4096 rows)")]:
        rows = [r for r in vt if r["op"] == op]
        impls = list(dict.fromkeys(r["impl"] for r in rows))  # first-seen order, torch last
        series = {i.removeprefix("ours:"): [(int(r["shape"].split("x")[1]), float(r["metric"]))
                                            for r in rows if r["impl"] == i] for i in impls}
        fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
        _lines(ax, series, xs_label=lambda x: f"{x // 1024}K" if x >= 1024 else str(x))
        if peak:
            ax.axhline(peak, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
            ax.annotate(f"theoretical peak ~{peak:.0f} GB/s", (ax.get_xlim()[0], peak), xytext=(4, 4),
                        textcoords="offset points", fontsize=8.5, color=INK_2)
        ax.set_ylim(0, (peak or max(y for s in series.values() for _, y in s)) * 1.12)
        title = "Softmax" if op == "softmax" else "LayerNorm"
        _style(ax, f"{title} effective bandwidth, {gpu}", xlabel, "GB/s (read input + write output once)")
        ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.12, 1))
        fig.tight_layout()
        fig.savefig(PLOTS / f"{op}.png", dpi=150)
        plt.close(fig)


def _peak_bw():
    rows = read_csv("softmax_cuda.csv")
    r = rows[0]
    return float(r["gbps"]) / (float(r["pct_peak_bw"]) / 100) if float(r["pct_peak_bw"]) else None


def _table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(headers))) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def write_results_md(env: dict):
    mm = read_csv("matmul_cuda.csv")
    npy = {int(r["N"]): float(r["gflops"]) for r in read_csv("matmul_numpy.csv")}
    sizes = sorted({int(r["N"]) for r in mm})
    variants = list(dict.fromkeys(r["variant"] for r in mm))
    g = {(r["variant"], int(r["N"])): float(r["gflops"]) for r in mm}
    ms = {(r["variant"], int(r["N"])): float(r["ms"]) for r in mm}
    mm_rows = [[v] + [f"{g[(v, n)]:,.0f}" for n in sizes] for v in variants]
    mm_rows.append(["numpy (CPU)"] + [f"{npy[n]:,.0f}" if n in npy else "-" for n in sizes])
    mm_ms_rows = [[v] + [f"{ms[(v, n)]:.3f}" for n in sizes] for v in variants]

    def bw_table(name, key):
        rows = read_csv(name)
        xs = sorted({int(r[key]) for r in rows})
        vs = list(dict.fromkeys(r["variant"] for r in rows))
        d = {(r["variant"], int(r[key])): (float(r["gbps"]), float(r["pct_peak_bw"])) for r in rows}
        return _table(["variant"] + [str(x) for x in xs],
                      [[v] + [f"{d[(v, x)][0]:.0f} ({d[(v, x)][1]:.0f}%)" for x in xs] for v in vs])

    vt = read_csv("vs_torch.csv")
    var = env.get("variation", {"rounds": 1, "median_spread_pct": "?", "p90_spread_pct": "?",
                                "max_spread_pct": "?"})
    tel = env.get("telemetry", {})

    def vt_table(op):
        rows = [r for r in vt if r["op"] == op]
        shapes = list(dict.fromkeys(r["shape"] for r in rows))
        impls = list(dict.fromkeys(r["impl"] for r in rows))
        d = {(r["impl"], r["shape"]): float(r["metric"]) for r in rows}
        unit = rows[0]["unit"]
        return _table([f"impl ({unit})"] + shapes, [[i] + [f"{d[(i, s)]:,.0f}" for s in shapes] for i in impls])

    warning = ""
    if not env.get("power_limit_ok", True):
        warning = (f"\n> **Warning: these results are not representative.** The laptop lowered the "
                   f"GPU power limit to {tel.get('power_limit_w_min')} W during the run (default "
                   f"{env['gpu_power_default_limit']}), so later measurements ran at reduced clocks.\n")
    md = f"""# Benchmark results

Generated by `python/run_benchmarks.py` from the CSVs in [`results/`](results/).
Every number below comes from that run; nothing is hand-edited.
{warning}
## Environment

| | |
|---|---|
| GPU | {env['gpu']} ({env['gpu_memory']}) |
| Driver / toolkit | {env['driver']} / {env['cuda_toolkit']} |
| GPU power limit during run | {tel.get('power_limit_w_min', '?')}-{tel.get('power_limit_w_max', '?')} W (default {env.get('gpu_power_default_limit', '?')}, max {env['gpu_power_max_limit']}), power source: {env['power_source']} |
| CPU (NumPy baseline) | {env['cpu']} |
| Software | Python {env['python']}, NumPy {env['numpy']}, PyTorch {env['torch']} |
| OS | {env['os']} |
| Date | {env['date']} |

Method:

- Every kernel is checked against a reference before it is timed.
- Each measurement is the median of 5-30 CUDA-event timings (fewer for slow
  kernels), after a sustained warm-up so the GPU is at its boost clock, with
  the L2 cache flushed before every timed run.
- The whole suite ran {var['rounds']} times; each table cell is the fastest
  of those rounds. This was a normal desktop session (browser, chat and other
  apps open), and other work can only slow a kernel down, so the fastest
  round is the closest to the kernel's own time. Spread between rounds
  (slowest / fastest - 1): median {var['median_spread_pct']}%, 90th percentile
  {var['p90_spread_pct']}%, worst {var['max_spread_pct']}%. Raw rounds are in
  [`results/raw/`](results/raw/).
- GPU clock while busy: median {tel.get('sm_clock_mhz_median', '?')} MHz
  (range {tel.get('sm_clock_mhz_min', '?')}-{tel.get('sm_clock_mhz_max', '?')} MHz),
  max temperature {tel.get('temp_c_max', '?')} °C, sampled every 0.5 s.

## Matrix multiplication (FP32, C++ harness)

GFLOP/s = 2·N³ / time. cuBLAS runs in its default FP32 math mode (no TF32).

![matmul](plots/matmul.png)

{_table(["GFLOP/s"] + [f"N={n}" for n in sizes], mm_rows)}

Time in ms:

{_table(["ms"] + [f"N={n}" for n in sizes], mm_ms_rows)}

## Softmax (C++ harness, 4096 rows)

Effective bandwidth = (read x + write y) / time, in GB/s (% of theoretical peak).
Columns are the row length.

{bw_table("softmax_cuda.csv", "cols")}

## LayerNorm (C++ harness, 4096 rows)

Effective bandwidth = (read x + write y + read gamma, beta) / time, in GB/s (% of peak).
Columns are the hidden size.

{bw_table("layernorm_cuda.csv", "hidden")}

## Ours vs. PyTorch (Python harness)

Both sides are called from Python on the same tensors and timed the same way
(including output allocation and launch overhead), so these are slightly
lower than the C++ numbers for small, fast kernels.

![softmax](plots/softmax.png)

![layernorm](plots/layernorm.png)

### Matmul

{vt_table("matmul")}

### Softmax

{vt_table("softmax")}

### LayerNorm

{vt_table("layernorm")}
"""
    (ROOT / "docs" / "RESULTS.md").write_text(md, encoding="utf-8")
    print(f"wrote {ROOT / 'docs' / 'RESULTS.md'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plots-only", action="store_true")
    p.add_argument("--allow-battery", action="store_true",
                   help="run even on battery (laptop GPUs are power-capped there)")
    p.add_argument("--rounds", type=int, default=3, help="repeat everything, keep the fastest per cell")
    args = p.parse_args()
    if not args.plots_only:
        if on_battery() and not args.allow_battery:
            sys.exit("On battery power: the GPU is power-capped. Plug in, or pass --allow-battery.")
        run_all(args.rounds)
    env = json.loads((RESULTS / "env.json").read_text())
    make_plots(env)
    write_results_md(env)
