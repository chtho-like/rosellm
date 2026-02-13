/*
CUDA GEMM Interview Series v2 (C = A @ B)

File: gemm_5_cp_async_multistage.cu
Focus:
  - Use cp.async (SM80+) to overlap global->shared copies with compute.
  - Use multi-stage SMEM (3 stages by default) to keep copies in flight.

What to explain in an interview:
  1) What cp.async is: an async copy queue from gmem->smem.
  2) Why "stages" exist: you want to prefetch far enough ahead.
  3) What commit_group / wait_group mean: control the async pipeline.

Build (SM80+ required):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_80 gemm_5_cp_async_multistage.cu \
    -o gemm_5_cp_async_multistage
Run:
  ./gemm_5_cp_async_multistage [M] [N] [K] [device_id]
Example:
  ./gemm_5_cp_async_multistage 4096 4096 4096 0

Notes:
  - Uses float4 LD/ST -> requires N % 4 == 0 and K % 4 == 0.
  - This demo uses cp.async for B only to keep the code compact.
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 8;
constexpr int kTM = 8;
constexpr int kTN = 8;
constexpr int kPad = 8;

constexpr int kStages = 3;  // 2..4 is common.

static_assert(kStages >= 2 && kStages <= 4, "bad stage count");
static_assert((kBM % kTM) == 0, "bad TM");
static_assert((kBN % kTN) == 0, "bad TN");

constexpr int kBlockX = kBN / kTN;  // 16
constexpr int kBlockY = kBM / kTM;  // 16

constexpr int kDefaultM = 4096;
constexpr int kDefaultN = 4096;
constexpr int kDefaultK = 4096;

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

__device__ __forceinline__ float4 LoadFloat4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}

__device__ __forceinline__ void StoreFloat4(float* p, const float4& v) {
  *reinterpret_cast<float4*>(p) = v;
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
  static_assert(kN >= 0 && kN <= 7, "cp.async wait_group immediate out of range");
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kN));
}

#else

// This file is meant for SM80+. Keep a compile-time guard for older arch.
__device__ __forceinline__ void CpAsyncCg16B(std::uint32_t, const void*) {}
__device__ __forceinline__ void CpAsyncCommit() {}
template <int kN>
__device__ __forceinline__ void CpAsyncWaitGroup() {
  (void)kN;
}

#endif

__device__ __forceinline__ void SmemStoreATransposed(float (*s_a)[kBM + kPad],
                                                     int k4,
                                                     int m_row,
                                                     const float4& v) {
  s_a[k4 + 0][m_row] = v.x;
  s_a[k4 + 1][m_row] = v.y;
  s_a[k4 + 2][m_row] = v.z;
  s_a[k4 + 3][m_row] = v.w;
}

__global__ void GemmKernelCpAsyncStages(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ c,
                                        int m,
                                        int n,
                                        int k) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
  (void)a;
  (void)b;
  (void)c;
  (void)m;
  (void)n;
  (void)k;
  return;
#else
  __shared__ float s_a[kStages][kBK][kBM + kPad];
  __shared__ float s_b[kStages][kBK][kBN + kPad];

  const int block_row = static_cast<int>(blockIdx.y) * kBM;
  const int block_col = static_cast<int>(blockIdx.x) * kBN;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  float r_c[kTM][kTN];
#pragma unroll
  for (int i = 0; i < kTM; ++i) {
#pragma unroll
    for (int j = 0; j < kTN; ++j) {
      r_c[i][j] = 0.0f;
    }
  }

  const int load_a_row = tid / 2;       // 0..127
  const int load_a_k4 = (tid & 1) * 4;  // 0 or 4
  const int load_b_k = tid / 32;        // 0..7
  const int load_b_col4 = (tid & 31) * 4;  // 0..124

  const int num_tiles = (k + kBK - 1) / kBK;
  const int prefetch_tiles = (num_tiles < (kStages - 1)) ? num_tiles
                                                         : (kStages - 1);

  auto load_a_reg = [&](int k0) -> float4 {
    const int g_row = block_row + load_a_row;
    const int g_col = k0 + load_a_k4;
    if (g_row < m && (g_col + 3) < k) {
      return LoadFloat4(a + static_cast<size_t>(g_row) * static_cast<size_t>(k) +
                        static_cast<size_t>(g_col));
    }
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  };

  // Prime the pipeline: prefetch first (kStages-1) tiles.
  for (int t = 0; t < prefetch_tiles; ++t) {
    const int k0 = t * kBK;
    const float4 a_reg = load_a_reg(k0);
    SmemStoreATransposed(s_a[t], load_a_k4, load_a_row, a_reg);

    const int g_row = k0 + load_b_k;
    const int g_col = block_col + load_b_col4;
    if (g_row < k && (g_col + 3) < n) {
      const std::uint32_t dst =
          __cvta_generic_to_shared(&s_b[t][load_b_k][load_b_col4]);
      CpAsyncCg16B(dst, b + static_cast<size_t>(g_row) * static_cast<size_t>(n) +
                           static_cast<size_t>(g_col));
    } else {
      StoreFloat4(&s_b[t][load_b_k][load_b_col4],
                  make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    }
    CpAsyncCommit();
  }

  if (prefetch_tiles > 0) {
    CpAsyncWaitGroup<kStages - 2>();
    __syncthreads();
  }

  const int a_m_top = ty * (kTM / 2);
  const int a_m_bot = (kBM / 2) + ty * (kTM / 2);
  const int b_n_left = tx * (kTN / 2);
  const int b_n_right = (kBN / 2) + tx * (kTN / 2);

  // Main loop: for tile t, compute stage (t % kStages), prefetch tile
  // (t + kStages - 1) into stage ((t + kStages - 1) % kStages).
  for (int t = 0; t < num_tiles; ++t) {
    const int cur = t % kStages;
    const int pre = t + (kStages - 1);
    if (pre < num_tiles) {
      const int stage = pre % kStages;
      const int k0 = pre * kBK;
      const float4 a_reg = load_a_reg(k0);
      SmemStoreATransposed(s_a[stage], load_a_k4, load_a_row, a_reg);

      const int g_row = k0 + load_b_k;
      const int g_col = block_col + load_b_col4;
      if (g_row < k && (g_col + 3) < n) {
        const std::uint32_t dst =
            __cvta_generic_to_shared(&s_b[stage][load_b_k][load_b_col4]);
        CpAsyncCg16B(dst,
                     b + static_cast<size_t>(g_row) * static_cast<size_t>(n) +
                         static_cast<size_t>(g_col));
      } else {
        StoreFloat4(&s_b[stage][load_b_k][load_b_col4],
                    make_float4(0.0f, 0.0f, 0.0f, 0.0f));
      }
      CpAsyncCommit();
    }

    // Compute tile t.
#pragma unroll
    for (int kk = 0; kk < kBK; ++kk) {
      const float4 a_top = LoadFloat4(&s_a[cur][kk][a_m_top]);
      const float4 a_bot = LoadFloat4(&s_a[cur][kk][a_m_bot]);
      const float4 b_l = LoadFloat4(&s_b[cur][kk][b_n_left]);
      const float4 b_r = LoadFloat4(&s_b[cur][kk][b_n_right]);

      const float a_vals[8] = {a_top.x, a_top.y, a_top.z, a_top.w,
                               a_bot.x, a_bot.y, a_bot.z, a_bot.w};
      const float b_vals[8] = {b_l.x, b_l.y, b_l.z, b_l.w,
                               b_r.x, b_r.y, b_r.z, b_r.w};

#pragma unroll
      for (int i = 0; i < kTM; ++i) {
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
          r_c[i][j] = __fmaf_rn(a_vals[i], b_vals[j], r_c[i][j]);
        }
      }
    }

    // Ensure the next tile we are about to use is ready.
    CpAsyncWaitGroup<kStages - 2>();
    __syncthreads();
  }

  // Store results (same 4-quadrant mapping).
  const int col_left = block_col + tx * (kTN / 2);
  for (int i = 0; i < (kTM / 2); ++i) {
    const int row_top = block_row + ty * (kTM / 2) + i;
    const int row_bot = block_row + (kBM / 2) + ty * (kTM / 2) + i;

    if (row_top < m && (col_left + 3) < n) {
      float* c_ptr = c + static_cast<size_t>(row_top) * static_cast<size_t>(n) +
                     static_cast<size_t>(col_left);
      StoreFloat4(c_ptr, make_float4(r_c[i][0], r_c[i][1], r_c[i][2], r_c[i][3]));
      if ((col_left + (kBN / 2) + 3) < n) {
        StoreFloat4(c_ptr + (kBN / 2),
                    make_float4(r_c[i][4], r_c[i][5], r_c[i][6], r_c[i][7]));
      }
    }

    if (row_bot < m && (col_left + 3) < n) {
      float* c_ptr = c + static_cast<size_t>(row_bot) * static_cast<size_t>(n) +
                     static_cast<size_t>(col_left);
      StoreFloat4(
          c_ptr,
          make_float4(r_c[i + (kTM / 2)][0], r_c[i + (kTM / 2)][1],
                      r_c[i + (kTM / 2)][2], r_c[i + (kTM / 2)][3]));
      if ((col_left + (kBN / 2) + 3) < n) {
        StoreFloat4(
            c_ptr + (kBN / 2),
            make_float4(r_c[i + (kTM / 2)][4], r_c[i + (kTM / 2)][5],
                        r_c[i + (kTM / 2)][6], r_c[i + (kTM / 2)][7]));
      }
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

  const dim3 block(kBlockX, kBlockY);
  const dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM, 1);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  GemmKernelCpAsyncStages<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
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
  std::printf("stages:      %d\n", kStages);
  std::printf("time_ms:     %.3f\n", ms);
  std::printf("max_abs_err: %.6g\n", max_abs_err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return (max_abs_err <= 1e-2) ? 0 : 1;
}

