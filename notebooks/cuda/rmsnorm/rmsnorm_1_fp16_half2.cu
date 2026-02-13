/*
CUDA RMSNorm Interview Series (fp16 + half2)

File: rmsnorm_1_fp16_half2.cu
Focus:
  - fp16 input/weight/output with fp32 accumulation.
  - half2 vectorization when cols is even.
  - This is close to what real inference kernels do (minus epilogue fusion).

Interview tags:
  - Classic:     High
  - Importance:  Very High
  - Frequency:   High

Memorize (recommended):
  - YES. RMSNorm is simpler than LayerNorm, and half2 is an easy "win".

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 rmsnorm_1_fp16_half2.cu \
    -o rmsnorm_1_fp16_half2
Run:
  ./rmsnorm_1_fp16_half2 [rows] [cols] [device_id]
Example:
  ./rmsnorm_1_fp16_half2 1024 1024 0
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

void FillInputFp16(std::vector<half>* x, int rows, int cols) {
  x->resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const int v = (r * 131 + c * 17) % 31 - 15;
      const float f = static_cast<float>(v) * 0.01f;
      (*x)[static_cast<size_t>(r) * static_cast<size_t>(cols) +
           static_cast<size_t>(c)] = __float2half_rn(f);
    }
  }
}

void FillWeightFp16(std::vector<half>* w, int cols) {
  w->resize(static_cast<size_t>(cols));
  for (int c = 0; c < cols; ++c) {
    const float f = 1.0f + static_cast<float>(c % 7) * 0.01f;
    (*w)[static_cast<size_t>(c)] = __float2half_rn(f);
  }
}

void CpuRmsNormFromFp16(const std::vector<half>& x,
                        const std::vector<half>& w,
                        std::vector<float>* y,
                        int rows,
                        int cols) {
  y->assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
  for (int r = 0; r < rows; ++r) {
    double sumsq = 0.0;
    for (int c = 0; c < cols; ++c) {
      const double xv = static_cast<double>(__half2float(
          x[static_cast<size_t>(r) * static_cast<size_t>(cols) +
            static_cast<size_t>(c)]));
      sumsq += xv * xv;
    }
    const double mean_sq = sumsq / static_cast<double>(cols);
    const double inv_rms =
        1.0 / std::sqrt(mean_sq + static_cast<double>(kEps));
    for (int c = 0; c < cols; ++c) {
      const double xv = static_cast<double>(__half2float(
          x[static_cast<size_t>(r) * static_cast<size_t>(cols) +
            static_cast<size_t>(c)]));
      const double ww = static_cast<double>(__half2float(w[c]));
      (*y)[static_cast<size_t>(r) * static_cast<size_t>(cols) +
           static_cast<size_t>(c)] = static_cast<float>(xv * inv_rms * ww);
    }
  }
}

__device__ __forceinline__ float WarpReduceSum(float v) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(kFullWarpMask, v, offset);
  }
  return v;
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

__global__ void RmsNormKernelFp16(const half* __restrict__ x,
                                  const half* __restrict__ w,
                                  half* __restrict__ y,
                                  int rows,
                                  int cols) {
  __shared__ float smem[kBlockSize];

  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const size_t row_base =
      static_cast<size_t>(row) * static_cast<size_t>(cols);

  const bool use_half2 = ((cols % 2) == 0);
  float local_sumsq = 0.0f;

  if (use_half2) {
    const int cols2 = cols / 2;
    const half2* x2 = reinterpret_cast<const half2*>(x + row_base);
    for (int i = threadIdx.x; i < cols2; i += kBlockSize) {
      const float2 fv = __half22float2(x2[i]);
      local_sumsq += fv.x * fv.x + fv.y * fv.y;
    }
  } else {
    for (int c = threadIdx.x; c < cols; c += kBlockSize) {
      const float xv = __half2float(x[row_base + static_cast<size_t>(c)]);
      local_sumsq += xv * xv;
    }
  }

  const float sumsq = BlockReduceSum(smem, local_sumsq);
  const float mean_sq = sumsq / static_cast<float>(cols);
  const float inv_rms = rsqrtf(mean_sq + kEps);
  __syncthreads();

  if (use_half2) {
    const int cols2 = cols / 2;
    const half2* x2 = reinterpret_cast<const half2*>(x + row_base);
    const half2* w2 = reinterpret_cast<const half2*>(w);
    half2* y2 = reinterpret_cast<half2*>(y + row_base);
    for (int i = threadIdx.x; i < cols2; i += kBlockSize) {
      const float2 xv = __half22float2(x2[i]);
      const float2 ww = __half22float2(w2[i]);
      const float o0 = xv.x * inv_rms * ww.x;
      const float o1 = xv.y * inv_rms * ww.y;
      y2[i] = __floats2half2_rn(o0, o1);
    }
  } else {
    for (int c = threadIdx.x; c < cols; c += kBlockSize) {
      const float xv = __half2float(x[row_base + static_cast<size_t>(c)]);
      const float ww = __half2float(w[c]);
      y[row_base + static_cast<size_t>(c)] = __float2half_rn(xv * inv_rms * ww);
    }
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

  std::vector<half> h_x;
  std::vector<half> h_w;
  FillInputFp16(&h_x, rows, cols);
  FillWeightFp16(&h_w, cols);

  std::vector<float> h_y_cpu;
  CpuRmsNormFromFp16(h_x, h_w, &h_y_cpu, rows, cols);

  half* d_x = nullptr;
  half* d_w = nullptr;
  half* d_y = nullptr;
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t x_bytes = n * sizeof(half);
  const size_t w_bytes = static_cast<size_t>(cols) * sizeof(half);
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
  RmsNormKernelFp16<<<rows, kBlockSize>>>(d_x, d_w, d_y, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<half> h_y_gpu_half(n);
  CUDA_CHECK(cudaMemcpy(h_y_gpu_half.data(), d_y, x_bytes,
                        cudaMemcpyDeviceToHost));

  double max_abs_err = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double y_gpu = static_cast<double>(__half2float(h_y_gpu_half[i]));
    const double diff =
        std::abs(y_gpu - static_cast<double>(h_y_cpu[i]));
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

  return (max_abs_err <= 5e-3) ? 0 : 1;
}
