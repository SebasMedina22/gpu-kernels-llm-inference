# GPU Kernels for LLM Inference

> **Status: In progress.** This is a learning project, built step by step. Each
> milestone is its own commit. Every benchmark number in this repo comes from a
> run I executed, and the GPU it ran on is noted next to it.

CUDA C++ implementations of the core operations in transformer inference:
matrix multiplication, softmax and LayerNorm. Each kernel starts as a naive
version, then gets optimized one step at a time. Every version is checked
against a reference implementation before it is timed.

## Roadmap

- [x] **1. Setup**: CMake project, repo layout, device query tool, Colab notebook, `vector_add` warm-up kernel (builds and passes on the RTX 5050 Laptop with CUDA 13.4 and MSVC 2022)
- [x] **2. Matrix multiplication**: naive (uncoalesced and coalesced), shared-memory tiled, and register-tiled kernels, verified against a CPU reference and compared with cuBLAS and a NumPy CPU baseline ([kernels](src/matmul/matmul_kernels.cu), [benchmark](benchmarks/bench_matmul.cu))
- [x] **3. Softmax**: row-wise and numerically stable (handles -inf masks), naive vs. warp-level shuffle reduction vs. warp-level online softmax ([kernels](src/softmax/softmax_kernels.cu), [benchmark](benchmarks/bench_softmax.cu))
- [x] **4. LayerNorm**: one thread per row vs. one warp per row vs. one block per row with the row cached in shared memory; two-pass (numerically stable) variance ([kernels](src/layernorm/layernorm_kernels.cu), [benchmark](benchmarks/bench_layernorm.cu))
- [x] **5. Python bindings**: C API in a shared library, called with `ctypes` on PyTorch CUDA tensors (zero-copy, on PyTorch's current stream); tests against `torch.matmul`, `torch.softmax` and `F.layer_norm` ([bindings](python/llm_kernels.py), [tests](python/test_kernels.py), [benchmark](python/bench_vs_torch.py))
- [ ] **6. Benchmark suite**: CUDA-event timing over several sizes, results table and plot in `docs/`. *Harness done ([run_benchmarks.py](python/run_benchmarks.py)): 3 rounds, cold L2, GPU warm-up, power and clock telemetry. Waiting on a clean full-power run (see Results).*
- [ ] **7. Profiling notes**: Nsight Compute analysis of why each optimization helps

## Results

No published numbers yet. Every kernel passes its correctness checks, and
the benchmark suite runs end to end, but the first full runs on this laptop
were not clean: about 40 seconds into a run, the laptop lowered the GPU's
power limit from 95 W to 24 W (below its 55 W default), and clocks dropped
with it. The suite now logs the enforced power limit throughout and marks a
run invalid if it drops. Results will be published here from a run where it
stays at full power.

## Hardware

| Environment | GPU | Compute capability |
|---|---|---|
| Local (Windows 11, CUDA 13.4) | NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB, 20 SMs, ~384 GB/s theoretical | 12.0 (Blackwell) |
| Google Colab (optional) | NVIDIA Tesla T4, 16 GB | 7.5 (Turing) |

Full `device_query` output for the local GPU:
[`docs/device_query_rtx5050_laptop.txt`](docs/device_query_rtx5050_laptop.txt).

## Repository layout

```
src/
  common/       shared helpers (CUDA error checking, timers)
  tools/        device_query: prints the GPU limits that shape kernel design
  warmup/       vector_add: first kernel, verified against a CPU loop
  ...           one folder per kernel family (added per milestone)
benchmarks/     timing harnesses
python/         Python bindings and comparison scripts
docs/           results tables, plots, profiling notes
notebooks/      Colab notebook to build and run everything on a cloud GPU
```

## Building

Requirements: CUDA Toolkit 12.x or 13.x, CMake 3.24 or newer, and a host C++
compiler (MSVC on Windows, GCC on Linux).

```bash
cmake -B build                       # targets the GPU in this machine
cmake --build build --config Release
./build/device_query                 # Windows: build\Release\device_query.exe
./build/vector_add                   # prints PASSED if GPU matches CPU
```

To target a different GPU, pass its architecture, for example
`-DCMAKE_CUDA_ARCHITECTURES=75` for a T4.

### Python

The bindings need PyTorch with CUDA (any CUDA version your driver supports;
the kernels are compiled separately, so it doesn't have to match the toolkit).

```bash
python -m venv .venv
.venv/bin/pip install numpy matplotlib torch --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple   # Windows: .venv\Scripts\pip
cd python
python test_kernels.py        # every variant vs. PyTorch
python bench_vs_torch.py      # timings vs. PyTorch
```

Why `ctypes` and not a PyTorch C++ extension: an extension must be compiled
with the same CUDA major version PyTorch was built with, which is fragile on
Windows. A plain C API only needs device pointers and a stream handle, both
of which PyTorch exposes (`tensor.data_ptr()`, `torch.cuda.current_stream().cuda_stream`).

### Running on Google Colab

Open [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) in Colab and
select a GPU runtime (*Runtime → Change runtime type → T4 GPU*). The notebook
clones the repo, builds it and runs the programs.
