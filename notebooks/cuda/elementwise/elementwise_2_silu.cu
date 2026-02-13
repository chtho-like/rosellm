/*
CUDA Elementwise Interview Series

File: elementwise_2_silu.cu
Focus:
  - SiLU / Swish activation:
      silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

Interview tags:
  - Classic:     High
  - Importance:  High (SwiGLU uses SiLU)
  - Frequency:   High

Memorize:
  - YES. Very common in modern LLMs (SwiGLU).

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 elementwise_2_silu.cu \
    -o elementwise_2_silu
Run:
  ./elementwise_2_silu [n] [device_id]
Example:
  ./elementwise_2_silu 1048576 0
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kDefaultN = 1 << 20;

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t err__ = (expr);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err__));                                \
      std::exit(1);                                                           \
    }                                                                         \
  } while (false)

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

void FillInput(std::vector<float>* x, int n) {
  x->resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    (*x)[static_cast<size_t>(i)] = static_cast<float>((i * 17) % 31 - 15) * 0.1f;
  }
}

float CpuSilu(float x) {
  const float s = 1.0f / (1.0f + std::exp(-x));
  return x * s;
}

void CpuSiluVec(const std::vector<float>& x, std::vector<float>* y) {
  y->resize(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    (*y)[i] = CpuSilu(x[i]);
  }
}

__device__ __forceinline__ float Silu(float x) {
  const float s = 1.0f / (1.0f + __expf(-x));
  return x * s;
}

__global__ void SiluKernel(const float* __restrict__ x,
                           float* __restrict__ y,
                           int n) {
  const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride = static_cast<int>(gridDim.x) * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    y[i] = Silu(x[i]);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int n = (argc >= 2) ? ParseInt(argv[1], kDefaultN) : kDefaultN;
  const int device_id = (argc >= 3) ? ParseInt(argv[2], 0) : 0;

  if (n <= 0) {
    std::fprintf(stderr, "n must be > 0\n");
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(device_id));

  std::vector<float> h_x;
  FillInput(&h_x, n);
  std::vector<float> h_y_cpu;
  CpuSiluVec(h_x, &h_y_cpu);

  float* d_x = nullptr;
  float* d_y = nullptr;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  const int blocks = 256;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  SiluKernel<<<blocks, kBlockSize>>>(d_x, d_y, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<float> h_y_gpu(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  double max_abs_err = 0.0;
  for (int i = 0; i < n; ++i) {
    const double diff =
        std::abs(static_cast<double>(h_y_gpu[static_cast<size_t>(i)]) -
                 static_cast<double>(h_y_cpu[static_cast<size_t>(i)]));
    max_abs_err = std::max(max_abs_err, diff);
  }

  std::printf("n:           %d\n", n);
  std::printf("time_ms:     %.3f\n", ms);
  std::printf("max_abs_err: %.6g\n", max_abs_err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  return (max_abs_err <= 1e-6) ? 0 : 1;
}

