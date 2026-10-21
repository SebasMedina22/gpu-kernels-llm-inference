cmake_minimum_required(VERSION 3.24)

# Default to the GPU in this machine ("native" = detect at configure time).
# Override for another GPU, e.g. -DCMAKE_CUDA_ARCHITECTURES=75 for a Colab T4.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(llm_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# Shared headers (error-checking macros, timers, ...).
add_library(common INTERFACE)
target_include_directories(common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# --- Tools -------------------------------------------------------------------
add_executable(device_query src/tools/device_query.cu)
target_link_libraries(device_query PRIVATE common)

# --- Kernels (added milestone by milestone) ----------------------------------
add_executable(vector_add src/warmup/vector_add.cu)
target_link_libraries(vector_add PRIVATE common)

# All kernels in one static library, shared by the benchmarks and the
# Python bindings.
add_library(llm_kernels STATIC
    src/matmul/matmul_kernels.cu
    src/softmax/softmax_kernels.cu
    src/layernorm/layernorm_kernels.cu
)
target_link_libraries(llm_kernels PUBLIC common)
set_target_properties(llm_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Keep source line info so Nsight Compute can map metrics back to the code.
target_compile_options(llm_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# Shared library with a C API, loaded from Python via ctypes.
add_library(llm_kernels_py SHARED src/capi/capi.cu)
target_link_libraries(llm_kernels_py PRIVATE llm_kernels)

# --- Benchmarks --------------------------------------------------------------
add_executable(bench_matmul benchmarks/bench_matmul.cu)
target_link_libraries(bench_matmul PRIVATE llm_kernels CUDA::cublas)

add_executable(bench_softmax benchmarks/bench_softmax.cu)
target_link_libraries(bench_softmax PRIVATE llm_kernels)

add_executable(bench_layernorm benchmarks/bench_layernorm.cu)
target_link_libraries(bench_layernorm PRIVATE llm_kernels)
