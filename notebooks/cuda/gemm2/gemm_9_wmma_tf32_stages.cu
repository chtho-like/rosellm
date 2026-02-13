/*
CUDA GEMM Interview Series v2 (Tensor Cores via WMMA, TF32)

File: gemm_9_wmma_tf32_stages.cu
Focus:
  - SGEMM on Tensor Cores using TF32 (SM80+):
    C (fp32) = A (fp32->tf32) @ B (fp32->tf32)
  - Show the "real" tiling hierarchy:
    - Block tile: 128x128
    - Warp tile:  32x64 (each warp does 2x4 WMMA tiles)
    - WMMA tile:  16x16x8
  - cp.async (gmem->smem) + multi-stage pipeline (3 stages).
  - Block swizzle (grid.x factorization via grid.z) to improve L2 locality.
  - SMEM padding to reduce bank conflicts.

Build (SM80+ required):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_80 gemm_9_wmma_tf32_stages.cu \
    -o gemm_9_wmma_tf32_stages
Run:
  ./gemm_9_wmma_tf32_stages [M] [N] [K] [device_id]
Example:
  ./gemm_9_wmma_tf32_stages 4096 4096 4096 0
*/

#include <cuda_runtime.h>

#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// WMMA TF32 shape.
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 8;

// Block tile is 128x128.
constexpr int kWarpCountM = 4;   // warps in M dimension
constexpr int kWarpCountN = 2;   // warps in N dimension
constexpr int kWarpTileM = 2;    // WMMA tiles per warp in M
constexpr int kWarpTileN = 4;    // WMMA tiles per warp in N

constexpr int kBM = kWmmaM * kWarpCountM * kWarpTileM;  // 128
constexpr int kBN = kWmmaN * kWarpCountN * kWarpTileN;  // 128
constexpr int kBK = kWmmaK;                             // 8

constexpr int kWarpsPerBlock = kWarpCountM * kWarpCountN;  // 8

constexpr int kStages = 3;  // 2..4 are common.
constexpr int kAPad = 8;
constexpr int kBPad = 8;

constexpr int kDefaultM = 4096;
constexpr int kDefaultN = 4096;
constexpr int kDefaultK = 4096;
constexpr int kDefaultSwizzleStride = 8;

static_assert(kBM == 128 && kBN == 128, "expected 128x128 block tile");
static_assert(kStages >= 2 && kStages <= 4, "bad stage count");

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
        sum += static_cast<double>(
                   a[static_cast<size_t>(i) * static_cast<size_t>(k) +
                     static_cast<size_t>(kk)]) *
               static_cast<double>(
                   b[static_cast<size_t>(kk) * static_cast<size_t>(n) +
                     static_cast<size_t>(j)]);
      }
      (*c)[static_cast<size_t>(i) * static_cast<size_t>(n) +
           static_cast<size_t>(j)] = static_cast<float>(sum);
    }
  }
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

__device__ __forceinline__ void CpAsyncCg16B(std::uint32_t dst_smem,
                                             const void* src_gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(dst_smem), "l"(src_gmem));
}

__device__ __forceinline__ void CpAsyncCommit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int kN>
__device__ __forceinline__ void CpAsyncWaitGroup() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kN));
}

#else

__device__ __forceinline__ void CpAsyncCg16B(std::uint32_t, const void*) {}
__device__ __forceinline__ void CpAsyncCommit() {}
template <int kN>
__device__ __forceinline__ void CpAsyncWaitGroup() {
  (void)kN;
}

#endif

__global__ void WmmaTf32StagesKernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ c,
                                     int m,
                                     int n,
                                     int k,
                                     int grid_x_total) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
  (void)a;
  (void)b;
  (void)c;
  (void)m;
  (void)n;
  (void)k;
  (void)grid_x_total;
  return;
#else
  using namespace nvcuda::wmma;

  const int bx = static_cast<int>(blockIdx.x) +
                 static_cast<int>(blockIdx.z) * static_cast<int>(gridDim.x);
  if (bx >= grid_x_total) {
    return;
  }
  const int by = static_cast<int>(blockIdx.y);

  const int block_row = by * kBM;
  const int block_col = bx * kBN;

  __shared__ float s_a[kStages][kBM][kBK + kAPad];
  __shared__ float s_b[kStages][kBK][kBN + kBPad];

  const int tid = static_cast<int>(threadIdx.y) * 32 + static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;  // 0..7
  const int lane_id = tid % 32;
  (void)lane_id;

  const int warp_m = warp_id / kWarpCountN;  // 0..3
  const int warp_n = warp_id - warp_m * kWarpCountN;  // 0..1

  fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag[kWarpTileM][kWarpTileN];
#pragma unroll
  for (int i = 0; i < kWarpTileM; ++i) {
#pragma unroll
    for (int j = 0; j < kWarpTileN; ++j) {
      fill_fragment(c_frag[i][j], 0.0f);
    }
  }

  // 256 threads -> 256 float4 loads for A tile and B tile.
  const int load_a_m = tid / 2;        // 0..127
  const int load_a_k4 = (tid & 1) * 4; // 0 or 4
  const int load_b_k = tid / 32;       // 0..7
  const int load_b_n4 = (tid & 31) * 4; // 0..124

  const int num_k_tiles = k / kBK;

  // Small-K fallback: load all tiles, then compute (no pipeline).
  if (num_k_tiles <= (kStages - 1)) {
    for (int t = 0; t < num_k_tiles; ++t) {
      const int k0 = t * kBK;
      const int g_a_row = block_row + load_a_m;
      const int g_a_col = k0 + load_a_k4;
      const int g_b_row = k0 + load_b_k;
      const int g_b_col = block_col + load_b_n4;

      if (g_a_row < m && (g_a_col + 3) < k) {
        const std::uint32_t dst =
            __cvta_generic_to_shared(&s_a[t][load_a_m][load_a_k4]);
        CpAsyncCg16B(dst,
                     a + static_cast<size_t>(g_a_row) * static_cast<size_t>(k) +
                         static_cast<size_t>(g_a_col));
      } else {
        *reinterpret_cast<float4*>(&s_a[t][load_a_m][load_a_k4]) =
            make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }

      if (g_b_row < k && (g_b_col + 3) < n) {
        const std::uint32_t dst =
            __cvta_generic_to_shared(&s_b[t][load_b_k][load_b_n4]);
        CpAsyncCg16B(dst,
                     b + static_cast<size_t>(g_b_row) * static_cast<size_t>(n) +
                         static_cast<size_t>(g_b_col));
      } else {
        *reinterpret_cast<float4*>(&s_b[t][load_b_k][load_b_n4]) =
            make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
      CpAsyncCommit();
    }

    CpAsyncWaitGroup<0>();
    __syncthreads();

    for (int t = 0; t < num_k_tiles; ++t) {
      fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, precision::tf32, row_major>
          a_frag[kWarpTileM];
      fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, precision::tf32, row_major>
          b_frag[kWarpTileN];

#pragma unroll
      for (int i = 0; i < kWarpTileM; ++i) {
        const int smem_a_m = warp_m * (kWmmaM * kWarpTileM) + i * kWmmaM;
        load_matrix_sync(a_frag[i], &s_a[t][smem_a_m][0], kBK + kAPad);
      }

#pragma unroll
      for (int j = 0; j < kWarpTileN; ++j) {
        const int smem_b_n = warp_n * (kWmmaN * kWarpTileN) + j * kWmmaN;
        load_matrix_sync(b_frag[j], &s_b[t][0][smem_b_n], kBN + kBPad);
      }

#pragma unroll
      for (int i = 0; i < kWarpTileM; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpTileN; ++j) {
          mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
      }
    }

    // Store C tiles (row-major).
#pragma unroll
    for (int i = 0; i < kWarpTileM; ++i) {
#pragma unroll
      for (int j = 0; j < kWarpTileN; ++j) {
        const int out_row =
            block_row + warp_m * (kWmmaM * kWarpTileM) + i * kWmmaM;
        const int out_col =
            block_col + warp_n * (kWmmaN * kWarpTileN) + j * kWmmaN;
        store_matrix_sync(
            c + static_cast<size_t>(out_row) * static_cast<size_t>(n) +
                static_cast<size_t>(out_col),
            c_frag[i][j], n, mem_row_major);
      }
    }
    return;
  }

  // Prime: load first (kStages-1) tiles.
#pragma unroll
  for (int t = 0; t < (kStages - 1); ++t) {
    const int k0 = t * kBK;
    const int g_a_row = block_row + load_a_m;
    const int g_a_col = k0 + load_a_k4;
    const int g_b_row = k0 + load_b_k;
    const int g_b_col = block_col + load_b_n4;

    if (g_a_row < m && (g_a_col + 3) < k) {
      const std::uint32_t dst =
          __cvta_generic_to_shared(&s_a[t][load_a_m][load_a_k4]);
      CpAsyncCg16B(dst, a + static_cast<size_t>(g_a_row) * static_cast<size_t>(k) +
                           static_cast<size_t>(g_a_col));
    } else {
      *reinterpret_cast<float4*>(&s_a[t][load_a_m][load_a_k4]) =
          make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    if (g_b_row < k && (g_b_col + 3) < n) {
      const std::uint32_t dst =
          __cvta_generic_to_shared(&s_b[t][load_b_k][load_b_n4]);
      CpAsyncCg16B(dst, b + static_cast<size_t>(g_b_row) * static_cast<size_t>(n) +
                           static_cast<size_t>(g_b_col));
    } else {
      *reinterpret_cast<float4*>(&s_b[t][load_b_k][load_b_n4]) =
          make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    CpAsyncCommit();
  }

  CpAsyncWaitGroup<kStages - 2>();
  __syncthreads();

#pragma unroll
  for (int k_tile = (kStages - 1); k_tile < num_k_tiles; ++k_tile) {
    const int smem_sel = (k_tile + 1) % kStages;
    const int smem_sel_next = k_tile % kStages;
    const int k0 = k_tile * kBK;

    // Prefetch tile k_tile into smem_sel_next.
    const int g_a_row = block_row + load_a_m;
    const int g_a_col = k0 + load_a_k4;
    const int g_b_row = k0 + load_b_k;
    const int g_b_col = block_col + load_b_n4;

    if (g_a_row < m && (g_a_col + 3) < k) {
      const std::uint32_t dst = __cvta_generic_to_shared(
          &s_a[smem_sel_next][load_a_m][load_a_k4]);
      CpAsyncCg16B(dst,
                   a + static_cast<size_t>(g_a_row) * static_cast<size_t>(k) +
                       static_cast<size_t>(g_a_col));
    } else {
      *reinterpret_cast<float4*>(&s_a[smem_sel_next][load_a_m][load_a_k4]) =
          make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    if (g_b_row < k && (g_b_col + 3) < n) {
      const std::uint32_t dst = __cvta_generic_to_shared(
          &s_b[smem_sel_next][load_b_k][load_b_n4]);
      CpAsyncCg16B(dst,
                   b + static_cast<size_t>(g_b_row) * static_cast<size_t>(n) +
                       static_cast<size_t>(g_b_col));
    } else {
      *reinterpret_cast<float4*>(&s_b[smem_sel_next][load_b_k][load_b_n4]) =
          make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    CpAsyncCommit();

    // Compute tile (k_tile - (kStages - 1)) from smem_sel.
    fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, precision::tf32, row_major>
        a_frag[kWarpTileM];
    fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, precision::tf32, row_major>
        b_frag[kWarpTileN];

#pragma unroll
    for (int i = 0; i < kWarpTileM; ++i) {
      const int smem_a_m = warp_m * (kWmmaM * kWarpTileM) + i * kWmmaM;
      load_matrix_sync(a_frag[i], &s_a[smem_sel][smem_a_m][0], kBK + kAPad);
    }

#pragma unroll
    for (int j = 0; j < kWarpTileN; ++j) {
      const int smem_b_n = warp_n * (kWmmaN * kWarpTileN) + j * kWmmaN;
      load_matrix_sync(b_frag[j], &s_b[smem_sel][0][smem_b_n], kBN + kBPad);
    }

#pragma unroll
    for (int i = 0; i < kWarpTileM; ++i) {
      if ((i & 1) != 0) {
        for (int j = kWarpTileN - 1; j >= 0; --j) {
          mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
      } else {
#pragma unroll
        for (int j = 0; j < kWarpTileN; ++j) {
          mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
      }
    }

    CpAsyncWaitGroup<kStages - 2>();
    __syncthreads();
  }

  // Drain: compute the last (kStages-1) tiles still sitting in SMEM.
  if ((kStages - 2) > 0) {
    CpAsyncWaitGroup<0>();
    __syncthreads();
  }

#pragma unroll
  for (int t = 0; t < (kStages - 1); ++t) {
    const int smem_sel = (num_k_tiles - (kStages - 1) + t) % kStages;
    fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, precision::tf32, row_major>
        a_frag[kWarpTileM];
    fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, precision::tf32, row_major>
        b_frag[kWarpTileN];

#pragma unroll
    for (int i = 0; i < kWarpTileM; ++i) {
      const int smem_a_m = warp_m * (kWmmaM * kWarpTileM) + i * kWmmaM;
      load_matrix_sync(a_frag[i], &s_a[smem_sel][smem_a_m][0], kBK + kAPad);
    }

#pragma unroll
    for (int j = 0; j < kWarpTileN; ++j) {
      const int smem_b_n = warp_n * (kWmmaN * kWarpTileN) + j * kWmmaN;
      load_matrix_sync(b_frag[j], &s_b[smem_sel][0][smem_b_n], kBN + kBPad);
    }

#pragma unroll
    for (int i = 0; i < kWarpTileM; ++i) {
#pragma unroll
      for (int j = 0; j < kWarpTileN; ++j) {
        mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
      }
    }
  }

  // Store C tiles (row-major).
#pragma unroll
  for (int i = 0; i < kWarpTileM; ++i) {
#pragma unroll
    for (int j = 0; j < kWarpTileN; ++j) {
      const int out_row = block_row + warp_m * (kWmmaM * kWarpTileM) + i * kWmmaM;
      const int out_col = block_col + warp_n * (kWmmaN * kWarpTileN) + j * kWmmaN;
      store_matrix_sync(c + static_cast<size_t>(out_row) * static_cast<size_t>(n) +
                            static_cast<size_t>(out_col),
                        c_frag[i][j], n, mem_row_major);
    }
  }
#endif
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
  if ((m % kBM) != 0 || (n % kBN) != 0 || (k % kBK) != 0) {
    std::fprintf(stderr,
                 "This demo requires M %% %d == 0, N %% %d == 0, K %% %d == 0 "
                 "(got %d,%d,%d)\n",
                 kBM, kBN, kBK, m, n, k);
    return 2;
  }
  if ((n % 4) != 0 || (k % 4) != 0) {
    std::fprintf(stderr, "This demo requires N %% 4 == 0 and K %% 4 == 0\n");
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

  const int grid_x_total = n / kBN;
  const int grid_y = m / kBM;
  const int swizzle_stride = kDefaultSwizzleStride;
  const int grid_x = std::min(grid_x_total, swizzle_stride);
  const int grid_z = (grid_x_total + grid_x - 1) / grid_x;

  const dim3 block(32, kWarpsPerBlock, 1);
  const dim3 grid(static_cast<unsigned int>(grid_x),
                  static_cast<unsigned int>(grid_y),
                  static_cast<unsigned int>(grid_z));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  WmmaTf32StagesKernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k, grid_x_total);
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

  std::printf("M:              %d\n", m);
  std::printf("N:              %d\n", n);
  std::printf("K:              %d\n", k);
  std::printf("stages:         %d\n", kStages);
  std::printf("block_swizzle:  x=%d z=%d (total_x=%d)\n", grid_x, grid_z,
              grid_x_total);
  std::printf("time_ms:        %.3f\n", ms);
  std::printf("max_abs_err:    %.6g\n", max_abs_err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return (max_abs_err <= 5e-1) ? 0 : 1;
}
