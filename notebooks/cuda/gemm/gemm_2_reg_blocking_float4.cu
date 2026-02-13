/*
CUDA GEMM Interview Series (C = A @ B)

File: gemm_2_reg_blocking_float4.cu
Focus:
  - Shared-memory tiling + register blocking:
    each thread computes a 4x4 output tile in registers.
  - Use float4 loads from shared memory for the B tile (and optionally
    float4 stores to C when N % 4 == 0).

Interview tags:
  - Classic:     Very High
  - Importance:  Extremely High
  - Frequency:   Very High

Memorize (highly recommended):
  - YES. This is a strong "hand-writeable" GEMM that shows real skill:
    tiling + register blocking + vectorization.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 gemm_2_reg_blocking_float4.cu \
    -o gemm_2_reg_blocking_float4
Run:
  ./gemm_2_reg_blocking_float4 [M] [N] [K] [device_id]
Example:
  ./gemm_2_reg_blocking_float4 256 256 256 0

Notes (how to "scale" this toward expert level):
  - Increase block tiles (e.g., 128x128) and add double-buffering.
  - Use cp.async (SM80+) to overlap global->shared copies with compute.
  - Tensor cores (WMMA/MMA) for fp16/bf16/tf32.
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kBlockK = 16;

constexpr int kThreadTileM = 4;  // each thread computes 4 rows
constexpr int kThreadTileN = 4;  // and 4 cols

static_assert((kBlockM % kThreadTileM) == 0, "bad M tile");
static_assert((kBlockN % kThreadTileN) == 0, "bad N tile");

constexpr int kBlockX = kBlockN / kThreadTileN;  // 16
constexpr int kBlockY = kBlockM / kThreadTileM;  // 16

constexpr int kDefaultM = 256;
constexpr int kDefaultN = 256;
constexpr int kDefaultK = 256;

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

void FillMatrix(std::vector<float>* a, int rows, int cols, int seed) {
  a->resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const int v = (i * 131 + j * 17 + seed) % 23 - 11;
      (*a)[static_cast<size_t>(i) * static_cast<size_t>(cols) +
           static_cast<size_t>(j)] = static_cast<float>(v) * 0.01f;
    }
  }
}

void CpuGemm(const std::vector<float>& a,
             const std::vector<float>& b,
             std::vector<float>* c,
             int m,
             int n,
             int k) {
  c->assign(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int kk = 0; kk < k; ++kk) {
        const double av =
            static_cast<double>(a[static_cast<size_t>(i) *
                                      static_cast<size_t>(k) +
                                  static_cast<size_t>(kk)]);
        const double bv =
            static_cast<double>(b[static_cast<size_t>(kk) *
                                      static_cast<size_t>(n) +
                                  static_cast<size_t>(j)]);
        sum += av * bv;
      }
      (*c)[static_cast<size_t>(i) * static_cast<size_t>(n) +
           static_cast<size_t>(j)] = static_cast<float>(sum);
    }
  }
}

__global__ void GemmKernelRegBlock(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ c,
                                   int m,
                                   int n,
                                   int k) {
  __shared__ float a_tile[kBlockM][kBlockK];
  __shared__ float b_tile[kBlockK][kBlockN];

  const int block_row = static_cast<int>(blockIdx.y) * kBlockM;
  const int block_col = static_cast<int>(blockIdx.x) * kBlockN;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  constexpr int kThreads = kBlockX * kBlockY;

  float4 acc[kThreadTileM];
  for (int i = 0; i < kThreadTileM; ++i) {
    acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  const bool can_store_vec4 = ((n % 4) == 0);

  for (int k0 = 0; k0 < k; k0 += kBlockK) {
    // Load A tile: [kBlockM, kBlockK].
    for (int idx = tid; idx < (kBlockM * kBlockK); idx += kThreads) {
      const int i = idx / kBlockK;
      const int kk = idx - i * kBlockK;
      const int global_row = block_row + i;
      const int global_col = k0 + kk;
      a_tile[i][kk] =
          (global_row < m && global_col < k)
              ? a[static_cast<size_t>(global_row) * static_cast<size_t>(k) +
                  static_cast<size_t>(global_col)]
              : 0.0f;
    }

    // Load B tile: [kBlockK, kBlockN].
    for (int idx = tid; idx < (kBlockK * kBlockN); idx += kThreads) {
      const int kk = idx / kBlockN;
      const int j = idx - kk * kBlockN;
      const int global_row = k0 + kk;
      const int global_col = block_col + j;
      b_tile[kk][j] =
          (global_row < k && global_col < n)
              ? b[static_cast<size_t>(global_row) * static_cast<size_t>(n) +
                  static_cast<size_t>(global_col)]
              : 0.0f;
    }

    __syncthreads();

    const int thread_row0 = block_row + threadIdx.y * kThreadTileM;
    const int thread_col0 = block_col + threadIdx.x * kThreadTileN;

#pragma unroll
    for (int kk = 0; kk < kBlockK; ++kk) {
      const float4 b4 =
          *reinterpret_cast<const float4*>(&b_tile[kk][threadIdx.x *
                                                    kThreadTileN]);

#pragma unroll
      for (int i = 0; i < kThreadTileM; ++i) {
        const float a_val = a_tile[threadIdx.y * kThreadTileM + i][kk];
        acc[i].x += a_val * b4.x;
        acc[i].y += a_val * b4.y;
        acc[i].z += a_val * b4.z;
        acc[i].w += a_val * b4.w;
      }
    }

    __syncthreads();
  }

  const int thread_row0 = block_row + threadIdx.y * kThreadTileM;
  const int thread_col0 = block_col + threadIdx.x * kThreadTileN;

  for (int i = 0; i < kThreadTileM; ++i) {
    const int row = thread_row0 + i;
    if (row >= m) {
      continue;
    }
    const int col = thread_col0;
    if (col >= n) {
      continue;
    }
    float* c_row = c + static_cast<size_t>(row) * static_cast<size_t>(n) +
                   static_cast<size_t>(col);
    if (can_store_vec4 && (col + 3 < n)) {
      *reinterpret_cast<float4*>(c_row) = acc[i];
    } else {
      if (col + 0 < n) c_row[0] = acc[i].x;
      if (col + 1 < n) c_row[1] = acc[i].y;
      if (col + 2 < n) c_row[2] = acc[i].z;
      if (col + 3 < n) c_row[3] = acc[i].w;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int m = (argc >= 2) ? ParseInt(argv[1], kDefaultM) : kDefaultM;
  const int n = (argc >= 3) ? ParseInt(argv[2], kDefaultN) : kDefaultN;
  const int k = (argc >= 4) ? ParseInt(argv[3], kDefaultK) : kDefaultK;
  const int device_id = (argc >= 5) ? ParseInt(argv[4], 0) : 0;

  if (m <= 0 || n <= 0 || k <= 0) {
    std::fprintf(stderr, "M, N, K must be > 0\n");
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(device_id));

  std::vector<float> h_a;
  std::vector<float> h_b;
  FillMatrix(&h_a, m, k, /*seed=*/1);
  FillMatrix(&h_b, k, n, /*seed=*/2);

  std::vector<float> h_c_cpu;
  CpuGemm(h_a, h_b, &h_c_cpu, m, n, k);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, static_cast<size_t>(m) * static_cast<size_t>(k) *
                                 sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, static_cast<size_t>(k) * static_cast<size_t>(n) *
                                 sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m) * static_cast<size_t>(n) *
                                 sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(),
                        static_cast<size_t>(m) * static_cast<size_t>(k) *
                            sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(),
                        static_cast<size_t>(k) * static_cast<size_t>(n) *
                            sizeof(float),
                        cudaMemcpyHostToDevice));

  const dim3 block(kBlockX, kBlockY);
  const dim3 grid((n + kBlockN - 1) / kBlockN, (m + kBlockM - 1) / kBlockM);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  GemmKernelRegBlock<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<float> h_c_gpu(static_cast<size_t>(m) *
                             static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c,
                        static_cast<size_t>(m) * static_cast<size_t>(n) *
                            sizeof(float),
                        cudaMemcpyDeviceToHost));

  double max_abs_err = 0.0;
  for (size_t i = 0; i < h_c_gpu.size(); ++i) {
    const double diff = std::abs(static_cast<double>(h_c_gpu[i]) -
                                 static_cast<double>(h_c_cpu[i]));
    max_abs_err = std::max(max_abs_err, diff);
  }

  std::printf("M:           %d\n", m);
  std::printf("N:           %d\n", n);
  std::printf("K:           %d\n", k);
  std::printf("time_ms:     %.3f\n", ms);
  std::printf("max_abs_err: %.6g\n", max_abs_err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return (max_abs_err <= 1e-3) ? 0 : 1;
}

