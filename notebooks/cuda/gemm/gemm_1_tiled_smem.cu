/*
CUDA GEMM Interview Series (C = A @ B)

File: gemm_1_tiled_smem.cu
Focus:
  - Classic optimization over gemm_0:
    tile A and B into shared memory to reuse global loads.
  - Still easy to explain and hand-write in an interview.

Interview tags:
  - Classic:     Very High
  - Importance:  Extremely High
  - Frequency:   Very High

Memorize (recommended):
  - YES. This is the "standard" interview GEMM starting point.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 gemm_1_tiled_smem.cu \
    -o gemm_1_tiled_smem
Run:
  ./gemm_1_tiled_smem [M] [N] [K] [device_id]
Example:
  ./gemm_1_tiled_smem 256 256 256 0

Typical interviewer follow-ups:
  - Register blocking (each thread computes multiple outputs).
  - Vectorize loads/stores (float4).
  - Increase arithmetic intensity with bigger tiles.
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kTile = 16;
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

__global__ void GemmKernelTiled(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int m,
                                int n,
                                int k) {
  __shared__ float a_tile[kTile][kTile];
  __shared__ float b_tile[kTile][kTile];

  const int row = static_cast<int>(blockIdx.y) * kTile + threadIdx.y;
  const int col = static_cast<int>(blockIdx.x) * kTile + threadIdx.x;

  float sum = 0.0f;
  const int num_tiles = (k + kTile - 1) / kTile;
  for (int t = 0; t < num_tiles; ++t) {
    const int a_col = t * kTile + threadIdx.x;
    const int b_row = t * kTile + threadIdx.y;

    a_tile[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k)
            ? a[static_cast<size_t>(row) * static_cast<size_t>(k) +
                static_cast<size_t>(a_col)]
            : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n)
            ? b[static_cast<size_t>(b_row) * static_cast<size_t>(n) +
                static_cast<size_t>(col)]
            : 0.0f;
    __syncthreads();

    for (int kk = 0; kk < kTile; ++kk) {
      sum += a_tile[threadIdx.y][kk] * b_tile[kk][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[static_cast<size_t>(row) * static_cast<size_t>(n) +
      static_cast<size_t>(col)] = sum;
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

  const dim3 block(kTile, kTile);
  const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  GemmKernelTiled<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
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

  return (max_abs_err <= 1e-4) ? 0 : 1;
}

