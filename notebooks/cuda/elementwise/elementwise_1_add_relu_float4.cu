/*
CUDA Elementwise Interview Series

File: elementwise_1_add_relu_float4.cu
Focus:
  - Fused add + ReLU:
      y[i] = max(0, a[i] + b[i])
  - float4 vectorization when n % 4 == 0.

Interview tags:
  - Classic:     High
  - Importance:  High
  - Frequency:   High

Memorize (recommended):
  - YES. This shows two key ideas:
    fusion (fewer memory passes) + vectorization (higher bandwidth).

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 elementwise_1_add_relu_float4.cu \
    -o elementwise_1_add_relu_float4
Run:
  ./elementwise_1_add_relu_float4 [n] [device_id]
Example:
  ./elementwise_1_add_relu_float4 1048576 0
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

void FillInput(std::vector<float>* a, std::vector<float>* b, int n) {
  a->resize(static_cast<size_t>(n));
  b->resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    (*a)[static_cast<size_t>(i)] = static_cast<float>((i * 17) % 31 - 15) * 0.1f;
    (*b)[static_cast<size_t>(i)] = static_cast<float>((i * 13) % 29 - 14) * 0.1f;
  }
}

void CpuAddRelu(const std::vector<float>& a,
                const std::vector<float>& b,
                std::vector<float>* y) {
  y->resize(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    const float v = a[i] + b[i];
    (*y)[i] = (v > 0.0f) ? v : 0.0f;
  }
}

__device__ __forceinline__ float Relu(float x) {
  return (x > 0.0f) ? x : 0.0f;
}

__global__ void AddReluKernel(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ y,
                              int n) {
  const bool vec4 = ((n % 4) == 0);
  if (vec4) {
    const int n4 = n / 4;
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* y4 = reinterpret_cast<float4*>(y);
    const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int stride = static_cast<int>(gridDim.x) * blockDim.x;
    for (int i = tid; i < n4; i += stride) {
      const float4 av = a4[i];
      const float4 bv = b4[i];
      float4 out{};
      out.x = Relu(av.x + bv.x);
      out.y = Relu(av.y + bv.y);
      out.z = Relu(av.z + bv.z);
      out.w = Relu(av.w + bv.w);
      y4[i] = out;
    }
    return;
  }

  const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride = static_cast<int>(gridDim.x) * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    y[i] = Relu(a[i] + b[i]);
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

  std::vector<float> h_a;
  std::vector<float> h_b;
  FillInput(&h_a, &h_b, n);
  std::vector<float> h_y_cpu;
  CpuAddRelu(h_a, h_b, &h_y_cpu);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_y = nullptr;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  const int blocks = 256;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  AddReluKernel<<<blocks, kBlockSize>>>(d_a, d_b, d_y, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<float> h_y_gpu(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  double max_abs_err = 0.0;
  for (int i = 0; i < n; ++i) {
    const double diff = std::abs(static_cast<double>(h_y_gpu[static_cast<size_t>(i)]) -
                                 static_cast<double>(h_y_cpu[static_cast<size_t>(i)]));
    max_abs_err = std::max(max_abs_err, diff);
  }

  std::printf("n:           %d\n", n);
  std::printf("time_ms:     %.3f\n", ms);
  std::printf("max_abs_err: %.6g\n", max_abs_err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_y));

  return (max_abs_err == 0.0) ? 0 : 1;
}

