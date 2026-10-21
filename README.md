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
- [ ] **5. Python bindings**: call the kernels from Python and compare with PyTorch
- [ ] **6. Benchmark suite**: CUDA-event timing over several sizes, results table and plot in `docs/`
- [ ] **7. Profiling notes**: Nsight Compute analysis of why each optimization helps

## Results

No benchmarks yet. Results will appear here as each milestone is completed.

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

### Running on Google Colab

Open [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) in Colab and
select a GPU runtime (*Runtime → Change runtime type → T4 GPU*). The notebook
clones the repo, builds it and runs the programs.
