/*
CUDA GEMM Interview Series v2 (Tensor Cores via MMA PTX)

File: gemm_10_mma_m16n8k16_smem_swizzle.cu
Focus:
  - Go below WMMA: use MMA PTX (m16n8k16) + ldmatrix.
  - Demonstrate the 3 "deep interview" topics:
    (1) ldmatrix (SMEM->regs for Tensor Cores)
    (2) SMEM swizzle/permutation (avoid bank conflicts for ldmatrix)
    (3) shuffle-based collective store (reduce global stores)

Kernel:
  - HMMA16816: mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
  - Block tile: 128x128 output (8 warps, 256 threads)
  - Each warp computes a 64x32 output tile via 4x4 MMA tiles.

Build (SM80+ recommended; needs ldmatrix + mma.sync):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_80 \
    gemm_10_mma_m16n8k16_smem_swizzle.cu -o gemm_10_mma_m16n8k16_smem_swizzle
Run:
  ./gemm_10_mma_m16n8k16_smem_swizzle [M] [N] [K] [device_id]
Example:
  ./gemm_10_mma_m16n8k16_smem_swizzle 1024 1024 1024 0

Notes:
  - This demo requires M and N multiples of 128, and K multiple of 16.
  - Output is FP16 (accumulating in FP16), so accuracy is limited.
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;

constexpr int kMmaTileM = 4;   // 4 * 16 = 64 rows per warp
constexpr int kMmaTileN = 4;   // 4 * 8  = 32 cols per warp
constexpr int kWarpCountM = 2; // 2 warps => 128 rows per block
constexpr int kWarpCountN = 4; // 4 warps => 128 cols per block

constexpr int kBM = kWarpCountM * kMmaTileM * kMmaM;  // 128
constexpr int kBN = kWarpCountN * kMmaTileN * kMmaN;  // 128
constexpr int kBK = kMmaK;                            // 16

constexpr int kWarpsPerBlock = kWarpCountM * kWarpCountN;  // 8

constexpr int kAPad = 8;
constexpr int kBPad = 8;

constexpr int kDefaultM = 1024;
constexpr int kDefaultN = 1024;
constexpr int kDefaultK = 1024;

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

void FillMatrixFp16(std::vector<half>* a, int rows, int cols, int seed) {
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

void CpuGemmFp16ToFp16(const std::vector<half>& a,
                       const std::vector<half>& b,
                       std::vector<half>* c,
                       int m,
                       int n,
                       int k) {
  c->assign(static_cast<size_t>(m) * static_cast<size_t>(n),
            __float2half_rn(0.0f));
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int kk = 0; kk < k; ++kk) {
        sum += static_cast<double>(__half2float(
                   a[static_cast<size_t>(i) * static_cast<size_t>(k) +
                     static_cast<size_t>(kk)])) *
               static_cast<double>(__half2float(
                   b[static_cast<size_t>(kk) * static_cast<size_t>(n) +
                     static_cast<size_t>(j)]));
      }
      (*c)[static_cast<size_t>(i) * static_cast<size_t>(n) +
           static_cast<size_t>(j)] = __float2half_rn(static_cast<float>(sum));
    }
  }
}

// 128-bit load/store helper (8 half = 16 bytes).
__device__ __forceinline__ int4 LoadInt4(const half* p) {
  return *reinterpret_cast<const int4*>(p);
}

__device__ __forceinline__ void StoreInt4(half* p, const int4& v) {
  *reinterpret_cast<int4*>(p) = v;
}

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                     \
  asm volatile(                                                               \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"    \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                \
      : "r"(addr))

#define LDMATRIX_X2_T(R0, R1, addr)                                           \
  asm volatile(                                                               \
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"      \
      : "=r"(R0), "=r"(R1)                                                    \
      : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)           \
  asm volatile(                                                               \
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "                    \
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"                    \
      : "=r"(RD0), "=r"(RD1)                                                  \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), \
        "r"(RC1))

// A SMEM swizzle for BK=16: each row has 2 segments of 8 half.
// We swap the 0..7 and 8..15 segments for alternating groups of 4 rows.
__device__ __forceinline__ int SwizzleACol(int row, int col0_or_8) {
  const int swap = ((row >> 2) & 1) << 3;  // 0 or 8
  return col0_or_8 ^ swap;
}

__global__ void MmaGemm128(const half* __restrict__ a,
                           const half* __restrict__ b,
                           half* __restrict__ c,
                           int m,
                           int n,
                           int k) {
  __shared__ half s_a[kBM][kBK + kAPad];
  __shared__ half s_b[kBK][kBN + kBPad];

  const int block_row = static_cast<int>(blockIdx.y) * kBM;
  const int block_col = static_cast<int>(blockIdx.x) * kBN;

  const int tid = static_cast<int>(threadIdx.y) * 32 + static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;  // 0..7
  const int lane_id = tid % 32;  // 0..31

  const int warp_m = warp_id & 1;      // 0..1
  const int warp_n = warp_id >> 1;     // 0..3

  const int load_a_m = tid / 2;          // 0..127
  const int load_a_k8 = (tid & 1) * 8;   // 0 or 8
  const int load_b_k = tid / 16;         // 0..15
  const int load_b_n8 = (tid & 15) * 8;  // 0..120

  std::uint32_t acc[kMmaTileM][kMmaTileN][2];
#pragma unroll
  for (int i = 0; i < kMmaTileM; ++i) {
#pragma unroll
    for (int j = 0; j < kMmaTileN; ++j) {
      acc[i][j][0] = 0u;
      acc[i][j][1] = 0u;
    }
  }

  for (int k0 = 0; k0 < k; k0 += kBK) {
    // gmem -> smem (128-bit).
    {
      const int g_a_row = block_row + load_a_m;
      const int g_a_col = k0 + load_a_k8;
      const int4 v = LoadInt4(a + static_cast<size_t>(g_a_row) * static_cast<size_t>(k) +
                              static_cast<size_t>(g_a_col));
      const int swz_col = SwizzleACol(load_a_m, load_a_k8);
      StoreInt4(&s_a[load_a_m][swz_col], v);
    }
    {
      const int g_b_row = k0 + load_b_k;
      const int g_b_col = block_col + load_b_n8;
      const int4 v = LoadInt4(b + static_cast<size_t>(g_b_row) * static_cast<size_t>(n) +
                              static_cast<size_t>(g_b_col));
      StoreInt4(&s_b[load_b_k][load_b_n8], v);
    }

    __syncthreads();

    // SMEM -> regs via ldmatrix.
    std::uint32_t ra[kMmaTileM][4];
    std::uint32_t rb[kMmaTileN][2];

#pragma unroll
    for (int i = 0; i < kMmaTileM; ++i) {
      const int base_m = warp_m * (kMmaTileM * kMmaM) + i * kMmaM;  // 0..112
      const int lane_m = base_m + (lane_id & 15);
      const int lane_k8 = (lane_id >> 4) * 8;  // 0 or 8
      const int lane_k8_swz = SwizzleACol(lane_m, lane_k8);
      const std::uint32_t a_ptr =
          __cvta_generic_to_shared(&s_a[lane_m][lane_k8_swz]);
      LDMATRIX_X4(ra[i][0], ra[i][1], ra[i][2], ra[i][3], a_ptr);
    }

#pragma unroll
    for (int j = 0; j < kMmaTileN; ++j) {
      const int base_n = warp_n * (kMmaTileN * kMmaN) + j * kMmaN;  // 0..120
      const int lane_k = lane_id & 15;
      const std::uint32_t b_ptr =
          __cvta_generic_to_shared(&s_b[lane_k][base_n]);
      LDMATRIX_X2_T(rb[j][0], rb[j][1], b_ptr);
    }

    // MMA compute (warp swizzle: reverse j order for odd i).
#pragma unroll
    for (int i = 0; i < kMmaTileM; ++i) {
      if ((i & 1) != 0) {
        for (int j = kMmaTileN - 1; j >= 0; --j) {
          HMMA16816(acc[i][j][0], acc[i][j][1], ra[i][0], ra[i][1], ra[i][2],
                    ra[i][3], rb[j][0], rb[j][1], acc[i][j][0], acc[i][j][1]);
        }
      } else {
#pragma unroll
        for (int j = 0; j < kMmaTileN; ++j) {
          HMMA16816(acc[i][j][0], acc[i][j][1], ra[i][0], ra[i][1], ra[i][2],
                    ra[i][3], rb[j][0], rb[j][1], acc[i][j][0], acc[i][j][1]);
        }
      }
    }

    __syncthreads();
  }

  // Shuffle-based collective store:
  // 4 lanes cooperate to write 1 row (8 half = 16 bytes) via a single uint4 store.
  const unsigned mask = 0xffffffffu;
  if ((lane_id & 3) == 0) {
    const int r = lane_id >> 2;  // 0..7

#pragma unroll
    for (int i = 0; i < kMmaTileM; ++i) {
#pragma unroll
      for (int j = 0; j < kMmaTileN; ++j) {
        const int tile_row = block_row + warp_m * (kMmaTileM * kMmaM) + i * kMmaM;
        const int tile_col = block_col + warp_n * (kMmaTileN * kMmaN) + j * kMmaN;

        const int row0 = tile_row + r;
        const int row1 = tile_row + r + 8;
        const int col0 = tile_col;

        const std::uint32_t v0 = acc[i][j][0];
        const std::uint32_t v1 = __shfl_sync(mask, acc[i][j][0], lane_id + 1);
        const std::uint32_t v2 = __shfl_sync(mask, acc[i][j][0], lane_id + 2);
        const std::uint32_t v3 = __shfl_sync(mask, acc[i][j][0], lane_id + 3);
        const uint4 pack0 = make_uint4(v0, v1, v2, v3);

        const std::uint32_t w0 = acc[i][j][1];
        const std::uint32_t w1 = __shfl_sync(mask, acc[i][j][1], lane_id + 1);
        const std::uint32_t w2 = __shfl_sync(mask, acc[i][j][1], lane_id + 2);
        const std::uint32_t w3 = __shfl_sync(mask, acc[i][j][1], lane_id + 3);
        const uint4 pack1 = make_uint4(w0, w1, w2, w3);

        *reinterpret_cast<uint4*>(
            c + static_cast<size_t>(row0) * static_cast<size_t>(n) +
                static_cast<size_t>(col0)) = pack0;
        *reinterpret_cast<uint4*>(
            c + static_cast<size_t>(row1) * static_cast<size_t>(n) +
                static_cast<size_t>(col0)) = pack1;
      }
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
  if ((m % kBM) != 0 || (n % kBN) != 0 || (k % kBK) != 0) {
    std::fprintf(stderr,
                 "This demo requires M %% %d == 0, N %% %d == 0, K %% %d == 0 "
                 "(got %d,%d,%d)\n",
                 kBM, kBN, kBK, m, n, k);
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(device_id));

  std::vector<half> h_a;
  std::vector<half> h_b;
  FillMatrixFp16(&h_a, m, k, /*seed=*/1);
  FillMatrixFp16(&h_b, k, n, /*seed=*/2);

  std::vector<half> h_c_cpu;
  CpuGemmFp16ToFp16(h_a, h_b, &h_c_cpu, m, n, k);

  half* d_a = nullptr;
  half* d_b = nullptr;
  half* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, static_cast<size_t>(m) * static_cast<size_t>(k) *
                                 sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_b, static_cast<size_t>(k) * static_cast<size_t>(n) *
                                 sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m) * static_cast<size_t>(n) *
                                 sizeof(half)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(),
                        static_cast<size_t>(m) * static_cast<size_t>(k) *
                            sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(),
                        static_cast<size_t>(k) * static_cast<size_t>(n) *
                            sizeof(half),
                        cudaMemcpyHostToDevice));

  const dim3 block(32, kWarpsPerBlock, 1);
  const dim3 grid(static_cast<unsigned int>(n / kBN),
                  static_cast<unsigned int>(m / kBM), 1);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  MmaGemm128<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<half> h_c_gpu(static_cast<size_t>(m) * static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c,
                        static_cast<size_t>(m) * static_cast<size_t>(n) *
                            sizeof(half),
                        cudaMemcpyDeviceToHost));

  double max_abs_err = 0.0;
  for (size_t i = 0; i < h_c_gpu.size(); ++i) {
    const double diff =
        std::abs(static_cast<double>(__half2float(h_c_gpu[i])) -
                 static_cast<double>(__half2float(h_c_cpu[i])));
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

  return (max_abs_err <= 1.0) ? 0 : 1;
}
