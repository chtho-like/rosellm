/*
CUDA LayerNorm Interview Series

File: layernorm_1_fused_sumsq_float4.cu
Focus:
  - Fuse mean+variance into ONE pass by reducing:
      sum(x) and sum(x^2)
    then var = E[x^2] - (E[x])^2.
  - Vectorize loads/stores with float4 when cols % 4 == 0.
  - Use warp-shuffle based block reductions (fewer __syncthreads()).

Interview tags:
  - Classic:     Very High
  - Importance:  Very High
  - Frequency:   Very High

Memorize (recommended):
  - YES. This is a compact, strong LayerNorm answer for interviews.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 layernorm_1_fused_sumsq_float4.cu \
    -o layernorm_1_fused_sumsq_float4
Run:
  ./layernorm_1_fused_sumsq_float4 [rows] [cols] [device_id]
Example:
  ./layernorm_1_fused_sumsq_float4 1024 1024 0

Notes:
  - sum/sumsq is fast but can be less numerically stable than Welford.
    See layernorm_2_welford_one_pass.cu.
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

__global__ void LayerNormKernelFused(const float* __restrict__ x,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
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

  const bool vec4 = (cols % 4) == 0;
  float local_sum = 0.0f;
  float local_sumsq = 0.0f;

  if (vec4) {
    const int cols4 = cols / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x + row_base);
    for (int i = threadIdx.x; i < cols4; i += kBlockSize) {
      const float4 v = x4[i];
      local_sum += v.x + v.y + v.z + v.w;
      local_sumsq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
  } else {
    for (int c = threadIdx.x; c < cols; c += kBlockSize) {
      const float v = x[row_base + static_cast<size_t>(c)];
      local_sum += v;
      local_sumsq += v * v;
    }
  }

  const float sum = BlockReduceSum(smem, local_sum);
  const float sumsq = BlockReduceSum(smem, local_sumsq);

  const float inv_cols = 1.0f / static_cast<float>(cols);
  const float mean = sum * inv_cols;
  float var = sumsq * inv_cols - mean * mean;
  var = fmaxf(var, 0.0f);
  const float inv_std = rsqrtf(var + kEps);

  if (vec4) {
    const int cols4 = cols / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x + row_base);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    const float4* b4 = reinterpret_cast<const float4*>(beta);
    float4* y4 = reinterpret_cast<float4*>(y + row_base);
    for (int i = threadIdx.x; i < cols4; i += kBlockSize) {
      const float4 xv = x4[i];
      const float4 gv = g4[i];
      const float4 bv = b4[i];
      float4 out{};
      out.x = ((xv.x - mean) * inv_std) * gv.x + bv.x;
      out.y = ((xv.y - mean) * inv_std) * gv.y + bv.y;
      out.z = ((xv.z - mean) * inv_std) * gv.z + bv.z;
      out.w = ((xv.w - mean) * inv_std) * gv.w + bv.w;
      y4[i] = out;
    }
  } else {
    for (int c = threadIdx.x; c < cols; c += kBlockSize) {
      const float xv = x[row_base + static_cast<size_t>(c)];
      const float norm = (xv - mean) * inv_std;
      const float out =
          norm * gamma[static_cast<size_t>(c)] + beta[static_cast<size_t>(c)];
      y[row_base + static_cast<size_t>(c)] = out;
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
  LayerNormKernelFused<<<rows, kBlockSize>>>(d_x, d_gamma, d_beta, d_y, rows,
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

  return (max_abs_err <= 1e-3) ? 0 : 1;
}
