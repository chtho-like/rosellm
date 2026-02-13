/*
CUDA Reduction Interview Series (sum reduction over a float array)

File: reduce_0_interleaved_modulo.cu
Focus: Baseline shared-memory reduction with interleaved addressing and
       a modulo-based predicate (divergence + expensive modulo).

This matches a very common CUDA interview prompt and follow-up chain:
  - "Write a CUDA kernel to compute sum(x[i]) (parallel reduction)."
  - Follow-up: "Now optimize it step by step (divergence, bank conflicts,
    global loads, warp-level primitives, etc.)."

Build:
  nvcc -O3 -std=c++17 -lineinfo reduce_0_interleaved_modulo.cu \
    -o reduce_0_interleaved_modulo
Run:
  ./reduce_0_interleaved_modulo [num_elements] [device_id]
Example:
  ./reduce_0_interleaved_modulo 16777216 0
Notes:
  - The first run can include CUDA driver JIT compilation; run twice to
    compare timings.
  - References (searchable): NVIDIA "Optimizing Parallel Reduction in CUDA",
    CUDA Samples "reduction".
*/

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kItemsPerThread = 1;
constexpr size_t kDefaultNumElements = 1u << 24;

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t err__ = (expr);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err__));                                \
      std::exit(1);                                                           \
    }                                                                         \
  } while (false)

__host__ __device__ constexpr size_t CeilDivSizeT(size_t a, size_t b) {
  return (a + b - 1) / b;
}

size_t ParseSizeT(const char* s, size_t default_value) {
  if (s == nullptr) {
    return default_value;
  }
  char* end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s) {
    return default_value;
  }
  return static_cast<size_t>(v);
}

int ParseInt(const char* s, int default_value) {
  if (s == nullptr) {
    return default_value;
  }
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (end == s) {
    return default_value;
  }
  return static_cast<int>(v);
}

double CpuSum(const std::vector<float>& x) {
  double sum = 0.0;
  for (float v : x) {
    sum += static_cast<double>(v);
  }
  return sum;
}

// Step 0 kernel:
// - Interleaved addressing: active threads are selected by tid % (2*s) == 0.
// - Downsides: modulo is expensive, and warps are partially active (divergence).
__global__ void ReduceKernel(const float* __restrict__ in,
                             float* __restrict__ out,
                             size_t n) {
  __shared__ float sdata[kBlockSize];

  const unsigned int tid = threadIdx.x;
  const size_t idx = static_cast<size_t>(blockIdx.x) *
                         static_cast<size_t>(kBlockSize) +
                     static_cast<size_t>(tid);

  sdata[tid] = (idx < n) ? in[idx] : 0.0f;
  __syncthreads();

  for (unsigned int stride = 1; stride < kBlockSize; stride <<= 1) {
    if ((tid % (2 * stride)) == 0) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[blockIdx.x] = sdata[0];
  }
}

float* GpuReduceSumInplace(float* d_in, float* d_tmp, size_t n) {
  float* in = d_in;
  float* out = d_tmp;

  size_t remaining = n;
  const size_t elements_per_block =
      static_cast<size_t>(kBlockSize) * static_cast<size_t>(kItemsPerThread);

  while (remaining > 1) {
    const int blocks =
        static_cast<int>(CeilDivSizeT(remaining, elements_per_block));
    ReduceKernel<<<blocks, kBlockSize>>>(in, out, remaining);
    CUDA_CHECK(cudaGetLastError());
    remaining = static_cast<size_t>(blocks);
    std::swap(in, out);
  }

  return in;
}

}  // namespace

int main(int argc, char** argv) {
  const size_t num_elements =
      (argc >= 2) ? ParseSizeT(argv[1], kDefaultNumElements)
                  : kDefaultNumElements;
  const int device_id = (argc >= 3) ? ParseInt(argv[2], 0) : 0;

  if (num_elements == 0) {
    std::fprintf(stderr, "num_elements must be > 0\n");
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(device_id));

  std::vector<float> h_in(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    const int v = static_cast<int>(i % 3) - 1;  // -1, 0, 1 (exact in float)
    h_in[i] = static_cast<float>(v);
  }

  const double cpu_sum = CpuSum(h_in);

  float* d_in = nullptr;
  float* d_tmp = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, num_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tmp, num_elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), num_elements * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  float* d_result = GpuReduceSumInplace(d_in, d_tmp, num_elements);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  float gpu_sum_f = 0.0f;
  CUDA_CHECK(cudaMemcpy(&gpu_sum_f, d_result, sizeof(float),
                        cudaMemcpyDeviceToHost));
  const double gpu_sum = static_cast<double>(gpu_sum_f);

  const double abs_err = std::abs(gpu_sum - cpu_sum);
  const double rel_err =
      (cpu_sum == 0.0) ? abs_err : (abs_err / std::abs(cpu_sum));

  std::printf("num_elements: %zu\n", num_elements);
  std::printf("cpu_sum:      %.0f\n", cpu_sum);
  std::printf("gpu_sum:      %.0f\n", gpu_sum);
  std::printf("abs_err:      %.6g\n", abs_err);
  std::printf("rel_err:      %.6g\n", rel_err);
  std::printf("time_ms:      %.3f\n", ms);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_tmp));

  return (rel_err <= 1e-6) ? 0 : 1;
}
