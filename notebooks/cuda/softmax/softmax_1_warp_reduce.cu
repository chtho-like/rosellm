/*
CUDA Softmax Interview Series (row-wise softmax over a 2D float matrix)

File: softmax_1_warp_reduce.cu
Focus:
  - Same math as softmax_0 (stable softmax).
  - Replace the full shared-memory reduction loop with a warp-shuffle based
    block reduction (fewer __syncthreads()).

Interview tags:
  - Classic:     High
  - Importance:  Very High
  - Frequency:   High

Memorize:
  - Optional. Good to memorize the "warp reduce + shared warp partials"
    pattern because it appears everywhere (softmax, layernorm, attention).

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 softmax_1_warp_reduce.cu \
    -o softmax_1_warp_reduce
Run:
  ./softmax_1_warp_reduce [rows] [cols] [device_id]
Example:
  ./softmax_1_warp_reduce 1024 1024 0

Typical interviewer follow-up after this:
  - "Can you vectorize loads (float4) and reduce exp calls?"
  - "How do you handle masked softmax?"
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kDefaultRows = 1024;
constexpr int kDefaultCols = 1024;
constexpr unsigned int kFullWarpMask = 0xFFFF'FFFFu;

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
           static_cast<size_t>(c)] = static_cast<float>(v) * 0.1f;
    }
  }
}

void CpuSoftmax(const std::vector<float>& x,
                std::vector<float>* y,
                int rows,
                int cols) {
  y->assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
  for (int r = 0; r < rows; ++r) {
    double max_v = -INFINITY;
    for (int c = 0; c < cols; ++c) {
      max_v = std::max(
          max_v, static_cast<double>(x[static_cast<size_t>(r) *
                                          static_cast<size_t>(cols) +
                                      static_cast<size_t>(c)]));
    }
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      const double v =
          static_cast<double>(x[static_cast<size_t>(r) *
                                    static_cast<size_t>(cols) +
                                static_cast<size_t>(c)]) -
          max_v;
      sum += std::exp(v);
    }
    const double inv = (sum == 0.0) ? 0.0 : (1.0 / sum);
    for (int c = 0; c < cols; ++c) {
      const double v =
          static_cast<double>(x[static_cast<size_t>(r) *
                                    static_cast<size_t>(cols) +
                                static_cast<size_t>(c)]) -
          max_v;
      (*y)[static_cast<size_t>(r) * static_cast<size_t>(cols) +
           static_cast<size_t>(c)] = static_cast<float>(std::exp(v) * inv);
    }
  }
}

__device__ __forceinline__ float WarpReduceMax(float v) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(kFullWarpMask, v, offset));
  }
  return v;
}

__device__ __forceinline__ float WarpReduceSum(float v) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(kFullWarpMask, v, offset);
  }
  return v;
}

__device__ __forceinline__ float BlockReduceMax(float* smem, float v) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  constexpr int kNumWarps = kBlockSize / kWarpSize;

  v = WarpReduceMax(v);
  if (lane == 0) {
    smem[warp] = v;
  }
  __syncthreads();

  if (warp == 0) {
    float warp_v = (lane < kNumWarps) ? smem[lane] : -INFINITY;
    warp_v = WarpReduceMax(warp_v);
    if (lane == 0) {
      smem[0] = warp_v;
    }
  }
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ float BlockReduceSum(float* smem, float v) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  constexpr int kNumWarps = kBlockSize / kWarpSize;

  v = WarpReduceSum(v);
  if (lane == 0) {
    smem[warp] = v;
  }
  __syncthreads();

  if (warp == 0) {
    float warp_v = (lane < kNumWarps) ? smem[lane] : 0.0f;
    warp_v = WarpReduceSum(warp_v);
    if (lane == 0) {
      smem[0] = warp_v;
    }
  }
  __syncthreads();
  return smem[0];
}

__global__ void SoftmaxKernel(const float* __restrict__ x,
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

  float local_max = -INFINITY;
  for (int col = threadIdx.x; col < cols; col += kBlockSize) {
    local_max = fmaxf(local_max, x[row_base + static_cast<size_t>(col)]);
  }
  const float row_max = BlockReduceMax(smem, local_max);

  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += kBlockSize) {
    local_sum += __expf(x[row_base + static_cast<size_t>(col)] - row_max);
  }
  const float row_sum = BlockReduceSum(smem, local_sum);
  const float inv = (row_sum == 0.0f) ? 0.0f : (1.0f / row_sum);

  for (int col = threadIdx.x; col < cols; col += kBlockSize) {
    y[row_base + static_cast<size_t>(col)] =
        __expf(x[row_base + static_cast<size_t>(col)] - row_max) * inv;
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
  FillInput(&h_x, rows, cols);

  std::vector<float> h_y_cpu;
  CpuSoftmax(h_x, &h_y_cpu, rows, cols);

  float* d_x = nullptr;
  float* d_y = nullptr;
  const size_t bytes =
      static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  SoftmaxKernel<<<rows, kBlockSize>>>(d_x, d_y, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<float> h_y_gpu(static_cast<size_t>(rows) *
                             static_cast<size_t>(cols));
  CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  double max_abs_err = 0.0;
  double max_row_sum_err = 0.0;
  for (int r = 0; r < rows; ++r) {
    double row_sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      const size_t idx =
          static_cast<size_t>(r) * static_cast<size_t>(cols) +
          static_cast<size_t>(c);
      const double diff = std::abs(static_cast<double>(h_y_gpu[idx]) -
                                   static_cast<double>(h_y_cpu[idx]));
      max_abs_err = std::max(max_abs_err, diff);
      row_sum += static_cast<double>(h_y_gpu[idx]);
    }
    max_row_sum_err = std::max(max_row_sum_err, std::abs(row_sum - 1.0));
  }

  std::printf("rows:            %d\n", rows);
  std::printf("cols:            %d\n", cols);
  std::printf("time_ms:         %.3f\n", ms);
  std::printf("max_abs_err:     %.6g\n", max_abs_err);
  std::printf("max_row_sum_err: %.6g\n", max_row_sum_err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  const bool ok = (max_abs_err <= 1e-5) && (max_row_sum_err <= 1e-5);
  return ok ? 0 : 1;
}
