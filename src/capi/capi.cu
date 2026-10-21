// Plain C API over the kernels, built as a shared library (llm_kernels_py)
// and called from Python with ctypes (see python/llm_kernels.py).
//
// All pointers are DEVICE pointers (e.g. torch.Tensor.data_ptr()) and
// `stream` is a cudaStream_t (e.g. torch.cuda.current_stream().cuda_stream).
// Functions return a cudaError_t code: 0 on success.

#include <cstring>

#include "layernorm/layernorm.cuh"
#include "matmul/matmul.cuh"
#include "softmax/softmax.cuh"

#if defined(_WIN32)
#define LK_API extern "C" __declspec(dllexport)
#else
#define LK_API extern "C" __attribute__((visibility("default")))
#endif

LK_API int lk_matmul(int variant, const float* A, const float* B, float* C, int M, int N, int K,
                     void* stream) {
    if (variant < 0 || variant >= kNumMatmulVariants) return cudaErrorInvalidValue;
    if (M == 0 || N == 0) return cudaSuccess;  // empty output: a 0-sized grid is a launch error
    launch_matmul(static_cast<MatmulVariant>(variant), A, B, C, M, N, K,
                  static_cast<cudaStream_t>(stream));
    return cudaGetLastError();
}

LK_API int lk_softmax(int variant, const float* x, float* y, int rows, int cols, void* stream) {
    if (variant < 0 || variant >= kNumSoftmaxVariants) return cudaErrorInvalidValue;
    if (rows == 0 || cols == 0) return cudaSuccess;
    launch_softmax(static_cast<SoftmaxVariant>(variant), x, y, rows, cols,
                   static_cast<cudaStream_t>(stream));
    return cudaGetLastError();
}

LK_API int lk_layernorm(int variant, const float* x, const float* gamma, const float* beta, float* y,
                        int rows, int hidden, float eps, void* stream) {
    if (variant < 0 || variant >= kNumLayerNormVariants) return cudaErrorInvalidValue;
    if (rows == 0 || hidden == 0) return cudaSuccess;
    return launch_layernorm(static_cast<LayerNormVariant>(variant), x, gamma, beta, y, rows, hidden,
                            eps, static_cast<cudaStream_t>(stream));
}

// Variant names, so Python doesn't duplicate the enums. Returns nullptr when
// `variant` is out of range; `op` is "matmul", "softmax" or "layernorm".
LK_API const char* lk_variant_name(const char* op, int variant) {
    if (std::strcmp(op, "matmul") == 0 && variant >= 0 && variant < kNumMatmulVariants)
        return matmul_variant_name(static_cast<MatmulVariant>(variant));
    if (std::strcmp(op, "softmax") == 0 && variant >= 0 && variant < kNumSoftmaxVariants)
        return softmax_variant_name(static_cast<SoftmaxVariant>(variant));
    if (std::strcmp(op, "layernorm") == 0 && variant >= 0 && variant < kNumLayerNormVariants)
        return layernorm_variant_name(static_cast<LayerNormVariant>(variant));
    return nullptr;
}

LK_API const char* lk_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
