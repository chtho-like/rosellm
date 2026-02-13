/*
CUDA Elementwise Interview Series

File: elementwise_3_gelu_tanh_approx.cu
Focus:
  - GELU (tanh approximation), commonly used in Transformers:
      gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Interview tags:
  - Classic:     High
  - Importance:  Medium/High (GELU appears in many models)
  - Frequency:   Medium

Memorize:
  - Optional. If you're LLM-inference focused, SiLU/SwiGLU is more common.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 elementwise_3_gelu_tanh_approx.cu \
    -o elementwise_3_gelu_tanh_approx
Run:
  ./elementwise_3_gelu_tanh_approx [n] [device_id]
Example:
  ./elementwise_3_gelu_tanh_approx 1048576 0
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

constexpr float kHalf = 0.5f;
constexpr float kSqrt2OverPi = 0.7978845608028654f;  // sqrt(2/pi)
constexpr float kGeluC = 0.044715f;

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

float CpuGelu(float x) {
  const float x3 = x * x * x;
  const float u = kSqrt2OverPi * (x + kGeluC * x3);
  return kHalf * x * (1.0f + std::tanh(u));
}

void CpuGeluVec(const std::vector<float>& x, std::vector<float>* y) {
  y->resize(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    (*y)[i] = CpuGelu(x[i]);
  }
}

__device__ __forceinline__ float Gelu(float x) {
  const float x3 = x * x * x;
  const float u = kSqrt2OverPi * (x + kGeluC * x3);
  return kHalf * x * (1.0f + tanhf(u));
}

__global__ void GeluKernel(const float* __restrict__ x,
                           float* __restrict__ y,
                           int n) {
  const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride = static_cast<int>(gridDim.x) * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    y[i] = Gelu(x[i]);
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
  CpuGeluVec(h_x, &h_y_cpu);

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
  GeluKernel<<<blocks, kBlockSize>>>(d_x, d_y, n);
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

  return (max_abs_err <= 1e-5) ? 0 : 1;
}

