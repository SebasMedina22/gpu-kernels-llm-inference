#include "matmul/matmul.cuh"

namespace {

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// v0: naive, uncoalesced
// One thread per C element. threadIdx.x selects the ROW, so the 32 threads of
// a warp read A at addresses K floats apart and write C at addresses N floats
// apart: every thread touches a different memory segment. Kept as the
// "what not to do" baseline for the coalescing discussion.
// ---------------------------------------------------------------------------
__global__ void matmul_naive_uncoalesced(const float* A, const float* B, float* C,
                                         int M, int N, int K) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// v1: naive, coalesced
// Same math, but threadIdx.x selects the COLUMN. Within a warp, all threads
// share `row` (A[row*K + k] is one broadcast load) and read consecutive
// B[k*N + col] values (one coalesced load). Still, every thread reads a full
// row of A and column of B from global memory: 2K loads per 2K FLOPs.
// ---------------------------------------------------------------------------
__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// v2: shared-memory tiling
// A block computes a TILE x TILE patch of C. It walks along K in steps of
// TILE: each step, the block cooperatively loads one TILE x TILE tile of A
// and one of B into shared memory (each thread loads one element of each),
// then every thread does TILE multiply-adds out of shared memory.
// Each global value is now loaded once per block instead of once per thread,
// cutting global traffic by a factor of TILE.
// ---------------------------------------------------------------------------
constexpr int TILE = 32;

__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.f;
    for (int t = 0; t < K; t += TILE) {
        // Out-of-range elements are loaded as 0 so they add nothing.
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.f;
        __syncthreads();  // tiles fully loaded before anyone reads them

        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();  // everyone done reading before the next load overwrites
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// v3: register tiling (2D block tiling + 2D thread tiling)
// v2 is limited by shared-memory bandwidth: every FMA needs two shared loads.
// Here each thread computes a TM x TN patch of outputs held in registers.
// Per k step it loads TM values of A and TN values of B into registers and
// does TM*TN FMAs with them, so shared loads per FMA drop from 2 to
// (TM+TN)/(TM*TN) = 0.25 for 8x8.
// Block tile: BM x BN outputs, 256 threads, stepping along K by BK.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
matmul_register_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    // A is stored transposed (k-major) so that a thread's TM values of A for
    // one k are contiguous. +4 padding spreads the transposing stores across
    // shared-memory banks (without it, 8 threads hit the same bank).
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int thread_row = tid / (BN / TN);  // which TM-row strip of the block tile
    const int thread_col = tid % (BN / TN);  // which TN-column strip
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};
    float reg_a[TM];
    float reg_b[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Cooperative load of the BM x BK tile of A (consecutive threads read
        // consecutive k, i.e. consecutive addresses).
        for (int i = tid; i < BM * BK; i += NUM_THREADS) {
            const int r = i / BK, c = i % BK;
            const int gr = block_row + r, gc = k0 + c;
            As[c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        // Cooperative load of the BK x BN tile of B (fully coalesced rows).
        for (int i = tid; i < BK * BN; i += NUM_THREADS) {
            const int r = i / BN, c = i % BN;
            const int gr = k0 + r, gc = block_col + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) reg_a[m] = As[k][thread_row * TM + m];
            #pragma unroll
            for (int n = 0; n < TN; ++n) reg_b[n] = Bs[k][thread_col * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n) acc[m][n] += reg_a[m] * reg_b[n];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int gr = block_row + thread_row * TM + m;
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            const int gc = block_col + thread_col * TN + n;
            if (gr < M && gc < N) C[gr * N + gc] = acc[m][n];
        }
    }
}

}  // namespace

const char* matmul_variant_name(MatmulVariant v) {
    switch (v) {
        case MatmulVariant::NaiveUncoalesced: return "naive_uncoalesced";
        case MatmulVariant::Naive: return "naive";
        case MatmulVariant::Tiled: return "tiled_smem";
        case MatmulVariant::RegisterTiled: return "register_tiled";
    }
    return "unknown";
}

void launch_matmul(MatmulVariant v, const float* A, const float* B, float* C,
                   int M, int N, int K, cudaStream_t stream) {
    switch (v) {
        case MatmulVariant::NaiveUncoalesced: {
            dim3 block(32, 32);
            dim3 grid(ceil_div(M, 32), ceil_div(N, 32));
            matmul_naive_uncoalesced<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            break;
        }
        case MatmulVariant::Naive: {
            dim3 block(32, 32);
            dim3 grid(ceil_div(N, 32), ceil_div(M, 32));
            matmul_naive<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            break;
        }
        case MatmulVariant::Tiled: {
            dim3 block(TILE, TILE);
            dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
            matmul_tiled<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            break;
        }
        case MatmulVariant::RegisterTiled: {
            constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
            dim3 block((BM / TM) * (BN / TN));
            dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
            matmul_register_tiled<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            break;
        }
    }
}
