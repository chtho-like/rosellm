/*
CUDA GEMM Interview Series v2 (Tensor Cores via WMMA, FP16)

File: gemm_8_wmma_fp16_tensorcore.cu
Focus:
  - Use Tensor Cores through the WMMA API:
    C (fp32) = A (fp16, row-major) @ B (fp16, col-major).
  - Show "tile warps": 4 warps per block compute a 32x32 output tile.

Build (SM70+ for WMMA, but recommend SM80+):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_80 gemm_8_wmma_fp16_tensorcore.cu \
    -o gemm_8_wmma_fp16_tensorcore
Run:
  ./gemm_8_wmma_fp16_tensorcore [M] [N] [K] [device_id]
Example:
  ./gemm_8_wmma_fp16_tensorcore 256 256 256 0

Notes:
  - This demo requires M and N to be multiples of 32, and K multiple of 16.
  - WMMA is the "API level" view; deep interviews go further into MMA PTX,
    ldmatrix, SMEM swizzle, cp.async stages, and Hopper WGMMA.
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

constexpr int kWarpsPerBlock = 4;  // 2x2 warp tile
constexpr int kBlockTile = 2;      // 2 tiles per dim => 32x32 outputs per block

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
           static_cast<size_t>(j)] =
          __float2half_rn(static_cast<float>(v) * 0.01f);
    }
  }
}

// B is KxN in column-major: B(k, n) is stored at B[n * K + k].
void FillMatrixFp16ColMajorKxN(std::vector<half>* b, int k, int n, int seed) {
  b->resize(static_cast<size_t>(k) * static_cast<size_t>(n));
  for (int kk = 0; kk < k; ++kk) {
    for (int j = 0; j < n; ++j) {
      const int v = (kk * 131 + j * 17 + seed) % 23 - 11;
      (*b)[static_cast<size_t>(j) * static_cast<size_t>(k) +
           static_cast<size_t>(kk)] =
          __float2half_rn(static_cast<float>(v) * 0.01f);
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
        const double av = static_cast<double>(__half2float(
            a[static_cast<size_t>(i) * static_cast<size_t>(k) +
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

__global__ void WmmaGemmKernel4Warps(const half* __restrict__ a_rowmajor,
                                     const half* __restrict__ b_colmajor,
                                     float* __restrict__ c_rowmajor,
                                     int m,
                                     int n,
                                     int k) {
  using namespace nvcuda::wmma;

  const int warp_id = static_cast<int>(threadIdx.y);
  const int warp_tile_r = warp_id / kBlockTile;  // 0..1
  const int warp_tile_c = warp_id - warp_tile_r * kBlockTile;  // 0..1

  const int tile_row = static_cast<int>(blockIdx.y) * kBlockTile + warp_tile_r;
  const int tile_col = static_cast<int>(blockIdx.x) * kBlockTile + warp_tile_c;
  const int row = tile_row * kWmmaM;
  const int col = tile_col * kWmmaN;

  fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
  fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < k; k0 += kWmmaK) {
    fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, half, row_major> a_frag;
    fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, half, col_major> b_frag;

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
  if ((m % (kWmmaM * kBlockTile)) != 0 || (n % (kWmmaN * kBlockTile)) != 0 ||
      (k % kWmmaK) != 0) {
    std::fprintf(stderr,
                 "This demo requires M,N multiples of %d and K multiple of %d "
                 "(got %d,%d,%d)\n",
                 kWmmaM * kBlockTile, kWmmaK, m, n, k);
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

  const dim3 block(32, kWarpsPerBlock, 1);
  const dim3 grid(static_cast<unsigned int>(n / (kWmmaN * kBlockTile)),
                  static_cast<unsigned int>(m / (kWmmaM * kBlockTile)), 1);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  WmmaGemmKernel4Warps<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
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

