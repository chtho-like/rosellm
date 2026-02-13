/*
CUDA Softmax Interview Series (row-wise softmax over a 2D fp16 matrix)

File: softmax_4_fp16_half2.cu
Focus:
  - fp16 input/output softmax with fp32 math for stability.
  - Optional causal mask (decoder-only LLM).
  - Use half2 vectorization when cols is even.

Interview tags:
  - Classic:     High
  - Importance:  Very High
  - Frequency:   High

Memorize:
  - Optional. This is a good "extra credit" file showing half2 technique.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 softmax_4_fp16_half2.cu \
    -o softmax_4_fp16_half2
Run:
  ./softmax_4_fp16_half2 [rows] [cols] [causal(0|1)] [device_id]
Example:
  ./softmax_4_fp16_half2 1024 1024 1 0

Notes:
  - In real attention kernels, softmax is usually fused with the QK^T and
    (softmax * V) steps to avoid materializing the probability matrix.
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

void FillInputFp16(std::vector<half>* x, int rows, int cols) {
  x->resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const int v = (r * 131 + c * 17) % 31 - 15;
      const float f = static_cast<float>(v) * 0.1f;
      (*x)[static_cast<size_t>(r) * static_cast<size_t>(cols) +
           static_cast<size_t>(c)] = __float2half_rn(f);
    }
  }
}

void CpuSoftmaxCausalFromFp16(const std::vector<half>& x,
                              std::vector<float>* y,
                              int rows,
                              int cols,
                              bool causal) {
  y->assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
  for (int r = 0; r < rows; ++r) {
    double max_v = -INFINITY;
    bool any = false;
    for (int c = 0; c < cols; ++c) {
      if (causal && (c > r)) {
        continue;
      }
      any = true;
      max_v = std::max(
          max_v, static_cast<double>(__half2float(
                     x[static_cast<size_t>(r) * static_cast<size_t>(cols) +
                       static_cast<size_t>(c)])));
    }
    if (!any) {
      continue;
    }
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      if (causal && (c > r)) {
        continue;
      }
      const double v =
          static_cast<double>(__half2float(
              x[static_cast<size_t>(r) * static_cast<size_t>(cols) +
                static_cast<size_t>(c)])) -
          max_v;
      sum += std::exp(v);
    }
    const double inv = (sum == 0.0) ? 0.0 : (1.0 / sum);
    for (int c = 0; c < cols; ++c) {
      if (causal && (c > r)) {
        (*y)[static_cast<size_t>(r) * static_cast<size_t>(cols) +
             static_cast<size_t>(c)] = 0.0f;
        continue;
      }
      const double v =
          static_cast<double>(__half2float(
              x[static_cast<size_t>(r) * static_cast<size_t>(cols) +
                static_cast<size_t>(c)])) -
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

__global__ void SoftmaxKernelFp16(const half* __restrict__ x,
                                  half* __restrict__ y,
                                  int rows,
                                  int cols,
                                  bool causal) {
  __shared__ float smem[kBlockSize];

  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  const size_t row_base =
      static_cast<size_t>(row) * static_cast<size_t>(cols);

  const bool use_half2 = ((cols % 2) == 0);
  float local_max = -INFINITY;

  if (use_half2) {
    const int cols2 = cols / 2;
    const half2* x2 = reinterpret_cast<const half2*>(x + row_base);
    for (int i = threadIdx.x; i < cols2; i += kBlockSize) {
      const int col0 = i * 2;
      const half2 hv = x2[i];
      const float2 fv = __half22float2(hv);
      if (!causal || (col0 + 0 <= row)) {
        local_max = fmaxf(local_max, fv.x);
      }
      if (!causal || (col0 + 1 <= row)) {
        local_max = fmaxf(local_max, fv.y);
      }
    }
  } else {
    for (int col = threadIdx.x; col < cols; col += kBlockSize) {
      if (causal && (col > row)) {
        continue;
      }
      local_max =
          fmaxf(local_max,
                __half2float(x[row_base + static_cast<size_t>(col)]));
    }
  }

  const float row_max = BlockReduceMax(smem, local_max);

  float local_sum = 0.0f;
  if (use_half2) {
    const int cols2 = cols / 2;
    const half2* x2 = reinterpret_cast<const half2*>(x + row_base);
    for (int i = threadIdx.x; i < cols2; i += kBlockSize) {
      const int col0 = i * 2;
      const half2 hv = x2[i];
      const float2 fv = __half22float2(hv);
      if (!causal || (col0 + 0 <= row)) {
        local_sum += __expf(fv.x - row_max);
      }
      if (!causal || (col0 + 1 <= row)) {
        local_sum += __expf(fv.y - row_max);
      }
    }
  } else {
    for (int col = threadIdx.x; col < cols; col += kBlockSize) {
      if (causal && (col > row)) {
        continue;
      }
      local_sum += __expf(
          __half2float(x[row_base + static_cast<size_t>(col)]) - row_max);
    }
  }

  const float row_sum = BlockReduceSum(smem, local_sum);
  const float inv = (row_sum == 0.0f) ? 0.0f : (1.0f / row_sum);

  if (use_half2) {
    const int cols2 = cols / 2;
    const half2* x2 = reinterpret_cast<const half2*>(x + row_base);
    half2* y2 = reinterpret_cast<half2*>(y + row_base);
    for (int i = threadIdx.x; i < cols2; i += kBlockSize) {
      const int col0 = i * 2;
      const half2 hv = x2[i];
      const float2 fv = __half22float2(hv);
      const float o0 =
          (causal && (col0 + 0 > row)) ? 0.0f : (__expf(fv.x - row_max) * inv);
      const float o1 =
          (causal && (col0 + 1 > row)) ? 0.0f : (__expf(fv.y - row_max) * inv);
      y2[i] = __floats2half2_rn(o0, o1);
    }
  } else {
    for (int col = threadIdx.x; col < cols; col += kBlockSize) {
      if (causal && (col > row)) {
        y[row_base + static_cast<size_t>(col)] = __float2half_rn(0.0f);
        continue;
      }
      const float v = __half2float(x[row_base + static_cast<size_t>(col)]);
      const float o = __expf(v - row_max) * inv;
      y[row_base + static_cast<size_t>(col)] = __float2half_rn(o);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int rows = (argc >= 2) ? ParseInt(argv[1], kDefaultRows) : kDefaultRows;
  const int cols = (argc >= 3) ? ParseInt(argv[2], kDefaultCols) : kDefaultCols;
  const bool causal = (argc >= 4) ? (ParseInt(argv[3], 0) != 0) : false;
  const int device_id = (argc >= 5) ? ParseInt(argv[4], 0) : 0;

  if (rows <= 0 || cols <= 0) {
    std::fprintf(stderr, "rows and cols must be > 0\n");
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(device_id));

  std::vector<half> h_x;
  FillInputFp16(&h_x, rows, cols);

  std::vector<float> h_y_cpu;
  CpuSoftmaxCausalFromFp16(h_x, &h_y_cpu, rows, cols, causal);

  half* d_x = nullptr;
  half* d_y = nullptr;
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t bytes = n * sizeof(half);
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  SoftmaxKernelFp16<<<rows, kBlockSize>>>(d_x, d_y, rows, cols, causal);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<half> h_y_gpu_half(n);
  CUDA_CHECK(cudaMemcpy(h_y_gpu_half.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  double max_abs_err = 0.0;
  double max_row_sum_err = 0.0;
  for (int r = 0; r < rows; ++r) {
    double row_sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      const size_t idx =
          static_cast<size_t>(r) * static_cast<size_t>(cols) +
          static_cast<size_t>(c);
      const double y_gpu = static_cast<double>(__half2float(h_y_gpu_half[idx]));
      const double diff = std::abs(y_gpu - static_cast<double>(h_y_cpu[idx]));
      max_abs_err = std::max(max_abs_err, diff);
      row_sum += y_gpu;
    }
    max_row_sum_err = std::max(max_row_sum_err, std::abs(row_sum - 1.0));
  }

  std::printf("rows:            %d\n", rows);
  std::printf("cols:            %d\n", cols);
  std::printf("causal:          %d\n", causal ? 1 : 0);
  std::printf("time_ms:         %.3f\n", ms);
  std::printf("max_abs_err:     %.6g\n", max_abs_err);
  std::printf("max_row_sum_err: %.6g\n", max_row_sum_err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  const bool ok = (max_abs_err <= 5e-3) && (max_row_sum_err <= 5e-3);
  return ok ? 0 : 1;
}
