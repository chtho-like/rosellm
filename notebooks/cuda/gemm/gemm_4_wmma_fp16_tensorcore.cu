/*
CUDA GEMM Interview Series (Tensor Cores via WMMA)

File: gemm_4_wmma_fp16_tensorcore.cu
Focus:
  - Use Tensor Cores through the WMMA API:
    C (fp32) = A (fp16, row-major) @ B (fp16, col-major).
  - One warp computes one 16x16 output tile (m16n16k16).

Interview tags:
  - Classic:     High
  - Importance:  Extremely High (LLM inference)
  - Frequency:   High (for senior GPU roles)

Memorize (optional but impressive):
  - YES (if you can). A minimal WMMA GEMM is a strong signal.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 gemm_4_wmma_fp16_tensorcore.cu \
    -o gemm_4_wmma_fp16_tensorcore
Run:
  ./gemm_4_wmma_fp16_tensorcore [M] [N] [K] [device_id]
Example:
  ./gemm_4_wmma_fp16_tensorcore 256 256 256 0

Notes:
  - WMMA uses fixed tile sizes. Real production kernels use MMA at a lower
    level, plus double-buffering, cp.async, epilogue fusion, etc.
  - Hopper (H100) introduces WGMMA for larger warp-group tiles.
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <mma.h>

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

void FillMatrixFp16RowMajor(std::vector<half>* a, int rows, int cols, int seed) {
  a->resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const int v = (i * 131 + j * 17 + seed) % 23 - 11;
      (*a)[static_cast<size_t>(i) * static_cast<size_t>(cols) +
           static_cast<size_t>(j)] = __float2half_rn(static_cast<float>(v) *
                                                     0.01f);
    }
  }
}

// B is KxN in column-major layout: B(k, n) is stored at B[n * K + k].
void FillMatrixFp16ColMajorKxN(std::vector<half>* b, int k, int n, int seed) {
  b->resize(static_cast<size_t>(k) * static_cast<size_t>(n));
  for (int kk = 0; kk < k; ++kk) {
    for (int j = 0; j < n; ++j) {
      const int v = (kk * 131 + j * 17 + seed) % 23 - 11;
      (*b)[static_cast<size_t>(j) * static_cast<size_t>(k) +
           static_cast<size_t>(kk)] = __float2half_rn(static_cast<float>(v) *
                                                      0.01f);
    }
  }
}

void CpuGemmFp16RowCol(const std::vector<half>& a,
                       const std::vector<half>& b_colmajor,
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
            static_cast<double>(__half2float(a[static_cast<size_t>(i) *
                                                static_cast<size_t>(k) +
                                              static_cast<size_t>(kk)]));
        const double bv = static_cast<double>(__half2float(
            b_colmajor[static_cast<size_t>(j) * static_cast<size_t>(k) +
                       static_cast<size_t>(kk)]));
        sum += av * bv;
      }
      (*c)[static_cast<size_t>(i) * static_cast<size_t>(n) +
           static_cast<size_t>(j)] = static_cast<float>(sum);
    }
  }
}

__global__ void WmmaGemmKernel(const half* __restrict__ a_rowmajor,
                               const half* __restrict__ b_colmajor,
                               float* __restrict__ c_rowmajor,
                               int m,
                               int n,
                               int k) {
  using namespace nvcuda::wmma;

  const int tile_row = static_cast<int>(blockIdx.y);
  const int tile_col = static_cast<int>(blockIdx.x);
  const int row = tile_row * kTile;
  const int col = tile_col * kTile;

  fragment<accumulator, kTile, kTile, kTile, float> c_frag;
  fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < k; k0 += kTile) {
    fragment<matrix_a, kTile, kTile, kTile, half, row_major> a_frag;
    fragment<matrix_b, kTile, kTile, kTile, half, col_major> b_frag;

    const half* a_ptr = a_rowmajor +
                        static_cast<size_t>(row) * static_cast<size_t>(k) +
                        static_cast<size_t>(k0);
    const half* b_ptr = b_colmajor +
                        static_cast<size_t>(col) * static_cast<size_t>(k) +
                        static_cast<size_t>(k0);

    load_matrix_sync(a_frag, a_ptr, k);
    load_matrix_sync(b_frag, b_ptr, k);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  store_matrix_sync(
      c_rowmajor + static_cast<size_t>(row) * static_cast<size_t>(n) +
          static_cast<size_t>(col),
      c_frag, n, mem_row_major);
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
  if ((m % kTile) != 0 || (n % kTile) != 0 || (k % kTile) != 0) {
    std::fprintf(stderr,
                 "WMMA demo requires M, N, K multiples of %d (got %d,%d,%d)\n",
                 kTile, m, n, k);
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(device_id));

  std::vector<half> h_a;
  std::vector<half> h_b_colmajor;
  FillMatrixFp16RowMajor(&h_a, m, k, /*seed=*/1);
  FillMatrixFp16ColMajorKxN(&h_b_colmajor, k, n, /*seed=*/2);

  std::vector<float> h_c_cpu;
  CpuGemmFp16RowCol(h_a, h_b_colmajor, &h_c_cpu, m, n, k);

  half* d_a = nullptr;
  half* d_b = nullptr;
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, static_cast<size_t>(m) * static_cast<size_t>(k) *
                                 sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_b, static_cast<size_t>(k) * static_cast<size_t>(n) *
                                 sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m) * static_cast<size_t>(n) *
                                 sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(),
                        static_cast<size_t>(m) * static_cast<size_t>(k) *
                            sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b_colmajor.data(),
                        static_cast<size_t>(k) * static_cast<size_t>(n) *
                            sizeof(half),
                        cudaMemcpyHostToDevice));

  const dim3 block(32, 1, 1);  // one warp
  const dim3 grid(static_cast<unsigned int>(n / kTile),
                  static_cast<unsigned int>(m / kTile), 1);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  WmmaGemmKernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
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

  return (max_abs_err <= 5e-2) ? 0 : 1;
}

