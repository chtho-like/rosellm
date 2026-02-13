/*
CUDA LayerNorm Interview Series (Welford one-pass)

File: layernorm_2_welford_one_pass.cu
Focus:
  - Compute mean and variance in ONE pass using Welford's algorithm
    (numerically stable).
  - Shows a "senior-level" skill: parallel Welford reduction.

Interview tags:
  - Classic:     Medium
  - Importance:  Very High
  - Frequency:   Medium (more common in senior interviews)

Memorize:
  - Optional. The idea matters more than memorizing every line.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 layernorm_2_welford_one_pass.cu \
    -o layernorm_2_welford_one_pass
Run:
  ./layernorm_2_welford_one_pass [rows] [cols] [device_id]
Example:
  ./layernorm_2_welford_one_pass 1024 1024 0
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kDefaultRows = 1024;
constexpr int kDefaultCols = 1024;
constexpr float kEps = 1e-5f;

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

void FillInput(std::vector<float>* x, int rows, int cols) {
  x->resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const int v = (r * 131 + c * 17) % 31 - 15;
      (*x)[static_cast<size_t>(r) * static_cast<size_t>(cols) +
           static_cast<size_t>(c)] = static_cast<float>(v) * 0.01f;
    }
  }
}

void FillGammaBeta(std::vector<float>* gamma,
                   std::vector<float>* beta,
                   int cols) {
  gamma->resize(static_cast<size_t>(cols));
  beta->resize(static_cast<size_t>(cols));
  for (int c = 0; c < cols; ++c) {
    (*gamma)[static_cast<size_t>(c)] = 1.0f + static_cast<float>(c % 7) * 0.01f;
    (*beta)[static_cast<size_t>(c)] = static_cast<float>((c % 5) - 2) * 0.01f;
  }
}

void CpuLayerNorm(const std::vector<float>& x,
                  const std::vector<float>& gamma,
                  const std::vector<float>& beta,
                  std::vector<float>* y,
                  int rows,
                  int cols) {
  y->assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
  for (int r = 0; r < rows; ++r) {
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      sum += static_cast<double>(
          x[static_cast<size_t>(r) * static_cast<size_t>(cols) +
            static_cast<size_t>(c)]);
    }
    const double mean = sum / static_cast<double>(cols);
    double var_sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      const double v = static_cast<double>(
                           x[static_cast<size_t>(r) *
                                 static_cast<size_t>(cols) +
                             static_cast<size_t>(c)]) -
                       mean;
      var_sum += v * v;
    }
    const double var = var_sum / static_cast<double>(cols);
    const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(kEps));
    for (int c = 0; c < cols; ++c) {
      const double xv = static_cast<double>(
          x[static_cast<size_t>(r) * static_cast<size_t>(cols) +
            static_cast<size_t>(c)]);
      const double norm = (xv - mean) * inv_std;
      const double out =
          norm * static_cast<double>(gamma[static_cast<size_t>(c)]) +
          static_cast<double>(beta[static_cast<size_t>(c)]);
      (*y)[static_cast<size_t>(r) * static_cast<size_t>(cols) +
           static_cast<size_t>(c)] = static_cast<float>(out);
    }
  }
}

struct WelfordData {
  float mean;
  float m2;
  int count;
};

__device__ __forceinline__ WelfordData WelfordCombine(WelfordData a,
                                                      WelfordData b) {
  if (a.count == 0) {
    return b;
  }
  if (b.count == 0) {
    return a;
  }
  const int count = a.count + b.count;
  const float delta = b.mean - a.mean;
  const float mean = a.mean + delta * (static_cast<float>(b.count) /
                                       static_cast<float>(count));
  const float m2 = a.m2 + b.m2 +
                   delta * delta *
                       (static_cast<float>(a.count) *
                        static_cast<float>(b.count) /
                        static_cast<float>(count));
  return WelfordData{mean, m2, count};
}

__device__ __forceinline__ WelfordData BlockReduceWelford(WelfordData v) {
  __shared__ WelfordData smem[kBlockSize];
  const int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] = WelfordCombine(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }
  return smem[0];
}

__global__ void LayerNormKernelWelford(const float* __restrict__ x,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float* __restrict__ y,
                                       int rows,
                                       int cols) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const size_t row_base =
      static_cast<size_t>(row) * static_cast<size_t>(cols);

  WelfordData local{0.0f, 0.0f, 0};
  for (int c = threadIdx.x; c < cols; c += kBlockSize) {
    const float xv = x[row_base + static_cast<size_t>(c)];
    local.count += 1;
    const float delta = xv - local.mean;
    local.mean += delta / static_cast<float>(local.count);
    const float delta2 = xv - local.mean;
    local.m2 += delta * delta2;
  }

  const WelfordData all = BlockReduceWelford(local);
  const float mean = all.mean;
  const float var =
      (all.count == 0) ? 0.0f : (all.m2 / static_cast<float>(all.count));
  const float inv_std = rsqrtf(var + kEps);

  for (int c = threadIdx.x; c < cols; c += kBlockSize) {
    const float xv = x[row_base + static_cast<size_t>(c)];
    const float norm = (xv - mean) * inv_std;
    const float out =
        norm * gamma[static_cast<size_t>(c)] + beta[static_cast<size_t>(c)];
    y[row_base + static_cast<size_t>(c)] = out;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int rows = (argc >= 2) ? ParseInt(argv[1], kDefaultRows) : kDefaultRows;
  const int cols = (argc >= 3) ? ParseInt(argv[2], kDefaultCols) : kDefaultCols;
  const int device_id = (argc >= 4) ? ParseInt(argv[3], 0) : 0;

  if (rows <= 0 || cols <= 0) {
    std::fprintf(stderr, "rows and cols must be > 0\n");
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(device_id));

  std::vector<float> h_x;
  std::vector<float> h_gamma;
  std::vector<float> h_beta;
  FillInput(&h_x, rows, cols);
  FillGammaBeta(&h_gamma, &h_beta, cols);

  std::vector<float> h_y_cpu;
  CpuLayerNorm(h_x, h_gamma, h_beta, &h_y_cpu, rows, cols);

  float* d_x = nullptr;
  float* d_gamma = nullptr;
  float* d_beta = nullptr;
  float* d_y = nullptr;
  const size_t x_bytes =
      static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
  const size_t w_bytes = static_cast<size_t>(cols) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
  CUDA_CHECK(cudaMalloc(&d_gamma, w_bytes));
  CUDA_CHECK(cudaMalloc(&d_beta, w_bytes));
  CUDA_CHECK(cudaMalloc(&d_y, x_bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_gamma, h_gamma.data(), w_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), w_bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  LayerNormKernelWelford<<<rows, kBlockSize>>>(d_x, d_gamma, d_beta, d_y, rows,
                                               cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<float> h_y_gpu(static_cast<size_t>(rows) *
                             static_cast<size_t>(cols));
  CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, x_bytes, cudaMemcpyDeviceToHost));

  double max_abs_err = 0.0;
  for (size_t i = 0; i < h_y_gpu.size(); ++i) {
    const double diff = std::abs(static_cast<double>(h_y_gpu[i]) -
                                 static_cast<double>(h_y_cpu[i]));
    max_abs_err = std::max(max_abs_err, diff);
  }

  std::printf("rows:        %d\n", rows);
  std::printf("cols:        %d\n", cols);
  std::printf("time_ms:     %.3f\n", ms);
  std::printf("max_abs_err: %.6g\n", max_abs_err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_beta));
  CUDA_CHECK(cudaFree(d_y));

  return (max_abs_err <= 1e-4) ? 0 : 1;
}

