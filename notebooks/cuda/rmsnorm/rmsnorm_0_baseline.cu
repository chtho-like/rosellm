/*
CUDA RMSNorm Interview Series

RMSNorm for X[rows, cols] (row-major):
  rms  = sqrt((1/cols) * sum_j x[j]^2 + eps)
  y[j] = x[j] / rms * weight[j]

File: rmsnorm_0_baseline.cu
Focus:
  - Correctness-first RMSNorm in fp32.
  - One CUDA block processes one row.

Interview tags:
  - Classic:     High
  - Importance:  Very High (common in LLMs: RMSNorm / ScaleNorm variants)
  - Frequency:   High

Memorize:
  - Baseline. The optimized version is rmsnorm_1_fp16_half2.cu.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 rmsnorm_0_baseline.cu \
    -o rmsnorm_0_baseline
Run:
  ./rmsnorm_0_baseline [rows] [cols] [device_id]
Example:
  ./rmsnorm_0_baseline 1024 1024 0
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
constexpr float kEps = 1e-6f;

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

void FillWeight(std::vector<float>* w, int cols) {
  w->resize(static_cast<size_t>(cols));
  for (int c = 0; c < cols; ++c) {
    (*w)[static_cast<size_t>(c)] = 1.0f + static_cast<float>(c % 7) * 0.01f;
  }
}

void CpuRmsNorm(const std::vector<float>& x,
                const std::vector<float>& w,
                std::vector<float>* y,
                int rows,
                int cols) {
  y->assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
  for (int r = 0; r < rows; ++r) {
    double sumsq = 0.0;
    for (int c = 0; c < cols; ++c) {
      const double xv = static_cast<double>(
          x[static_cast<size_t>(r) * static_cast<size_t>(cols) +
            static_cast<size_t>(c)]);
      sumsq += xv * xv;
    }
    const double mean_sq = sumsq / static_cast<double>(cols);
    const double inv_rms =
        1.0 / std::sqrt(mean_sq + static_cast<double>(kEps));
    for (int c = 0; c < cols; ++c) {
      const double xv = static_cast<double>(
          x[static_cast<size_t>(r) * static_cast<size_t>(cols) +
            static_cast<size_t>(c)]);
      const double out =
          xv * inv_rms * static_cast<double>(w[static_cast<size_t>(c)]);
      (*y)[static_cast<size_t>(r) * static_cast<size_t>(cols) +
           static_cast<size_t>(c)] = static_cast<float>(out);
    }
  }
}

__device__ __forceinline__ float BlockReduceSum(float* smem, float v) {
  const int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  return smem[0];
}

__global__ void RmsNormKernel(const float* __restrict__ x,
                              const float* __restrict__ w,
                              float* __restrict__ y,
                              int rows,
                              int cols) {
  __shared__ float smem[kBlockSize];

  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const size_t row_base =
      static_cast<size_t>(row) * static_cast<size_t>(cols);

  float local_sumsq = 0.0f;
  for (int c = threadIdx.x; c < cols; c += kBlockSize) {
    const float xv = x[row_base + static_cast<size_t>(c)];
    local_sumsq += xv * xv;
  }
  const float sumsq = BlockReduceSum(smem, local_sumsq);
  const float mean_sq = sumsq / static_cast<float>(cols);
  const float inv_rms = rsqrtf(mean_sq + kEps);
  __syncthreads();

  for (int c = threadIdx.x; c < cols; c += kBlockSize) {
    const float xv = x[row_base + static_cast<size_t>(c)];
    y[row_base + static_cast<size_t>(c)] = xv * inv_rms * w[c];
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
  std::vector<float> h_w;
  FillInput(&h_x, rows, cols);
  FillWeight(&h_w, cols);

  std::vector<float> h_y_cpu;
  CpuRmsNorm(h_x, h_w, &h_y_cpu, rows, cols);

  float* d_x = nullptr;
  float* d_w = nullptr;
  float* d_y = nullptr;
  const size_t x_bytes =
      static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
  const size_t w_bytes = static_cast<size_t>(cols) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
  CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
  CUDA_CHECK(cudaMalloc(&d_y, x_bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), w_bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  RmsNormKernel<<<rows, kBlockSize>>>(d_x, d_w, d_y, rows, cols);
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
  CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_y));

  return (max_abs_err <= 1e-4) ? 0 : 1;
}

