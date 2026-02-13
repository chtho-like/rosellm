/*
CUDA GEMM Interview Series v2 (C = A @ B)

File: gemm_6_splitk_workspace.cu
Focus:
  - Split-K parallelism without atomics:
    1) Kernel A: compute partial sums for each split over K into a workspace.
    2) Kernel B: reduce (sum) the workspace across split_k into final C.

Why this matters:
  - Split-K can increase parallelism when K is huge but M/N tiles are small.
  - Atomics are the "easy" interview answer; workspace reduction is the
    production-friendly pattern (deterministic and often faster).

Build:
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_80 gemm_6_splitk_workspace.cu \
    -o gemm_6_splitk_workspace
Run:
  ./gemm_6_splitk_workspace [M] [N] [K] [split_k] [device_id]
Example:
  ./gemm_6_splitk_workspace 512 512 8192 8 0
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;
constexpr int kDefaultM = 512;
constexpr int kDefaultN = 512;
constexpr int kDefaultK = 4096;
constexpr int kDefaultSplitK = 4;

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

// Kernel A: compute partial sums.
__global__ void GemmSplitKPartials(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ partials,
                                   int m,
                                   int n,
                                   int k,
                                   int split_k) {
  const int row = static_cast<int>(blockIdx.y) * kBlockY + threadIdx.y;
  const int col = static_cast<int>(blockIdx.x) * kBlockX + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }

  const int z = static_cast<int>(blockIdx.z);
  const int k_begin = (k * z) / split_k;
  const int k_end = (k * (z + 1)) / split_k;

  float sum = 0.0f;
  const size_t a_row_base =
      static_cast<size_t>(row) * static_cast<size_t>(k);
  for (int kk = k_begin; kk < k_end; ++kk) {
    sum += a[a_row_base + static_cast<size_t>(kk)] *
           b[static_cast<size_t>(kk) * static_cast<size_t>(n) +
             static_cast<size_t>(col)];
  }

  const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
  const size_t out_idx =
      static_cast<size_t>(z) * mn +
      static_cast<size_t>(row) * static_cast<size_t>(n) +
      static_cast<size_t>(col);
  partials[out_idx] = sum;
}

// Kernel B: reduce partial sums across split_k.
__global__ void SplitKReduce(const float* __restrict__ partials,
                             float* __restrict__ c,
                             int m,
                             int n,
                             int split_k) {
  const int row = static_cast<int>(blockIdx.y) * kBlockY + threadIdx.y;
  const int col = static_cast<int>(blockIdx.x) * kBlockX + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }

  const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
  const size_t base =
      static_cast<size_t>(row) * static_cast<size_t>(n) +
      static_cast<size_t>(col);

  float sum = 0.0f;
  for (int z = 0; z < split_k; ++z) {
    sum += partials[static_cast<size_t>(z) * mn + base];
  }
  c[base] = sum;
}

}  // namespace

int main(int argc, char** argv) {
  const int m = (argc >= 2) ? ParseInt(argv[1], kDefaultM) : kDefaultM;
  const int n = (argc >= 3) ? ParseInt(argv[2], kDefaultN) : kDefaultN;
  const int k = (argc >= 4) ? ParseInt(argv[3], kDefaultK) : kDefaultK;
  const int split_k =
      (argc >= 5) ? ParseInt(argv[4], kDefaultSplitK) : kDefaultSplitK;
  const int device_id = (argc >= 6) ? ParseInt(argv[5], 0) : 0;

  if (m <= 0 || n <= 0 || k <= 0 || split_k <= 0) {
    std::fprintf(stderr, "M, N, K, split_k must be > 0\n");
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
  float* d_partials = nullptr;
  float* d_c = nullptr;
  const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
  CUDA_CHECK(cudaMalloc(&d_a, static_cast<size_t>(m) * static_cast<size_t>(k) *
                                 sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, static_cast<size_t>(k) * static_cast<size_t>(n) *
                                 sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_partials, static_cast<size_t>(split_k) * mn *
                                         sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, mn * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(),
                        static_cast<size_t>(m) * static_cast<size_t>(k) *
                            sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(),
                        static_cast<size_t>(k) * static_cast<size_t>(n) *
                            sizeof(float),
                        cudaMemcpyHostToDevice));

  const dim3 block(kBlockX, kBlockY);
  const dim3 grid_partials((n + kBlockX - 1) / kBlockX,
                           (m + kBlockY - 1) / kBlockY,
                           static_cast<unsigned int>(split_k));
  const dim3 grid_reduce((n + kBlockX - 1) / kBlockX,
                         (m + kBlockY - 1) / kBlockY);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  GemmSplitKPartials<<<grid_partials, block>>>(d_a, d_b, d_partials, m, n, k,
                                               split_k);
  CUDA_CHECK(cudaGetLastError());
  SplitKReduce<<<grid_reduce, block>>>(d_partials, d_c, m, n, split_k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<float> h_c_gpu(mn);
  CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, mn * sizeof(float),
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
  std::printf("split_k:     %d\n", split_k);
  std::printf("time_ms:     %.3f\n", ms);
  std::printf("max_abs_err: %.6g\n", max_abs_err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_partials));
  CUDA_CHECK(cudaFree(d_c));

  return (max_abs_err <= 1e-2) ? 0 : 1;
}

