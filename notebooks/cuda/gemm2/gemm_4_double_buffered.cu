/*
CUDA GEMM Interview Series v2 (C = A @ B)

File: gemm_4_double_buffered.cu
Focus:
  - Add double-buffering on top of the SMEM layout from gemm_3:
    while computing tile t, preload tile t+1 (gmem->regs) and then
    write it to the other SMEM buffer.
  - Reduce per-tile synchronizations (pipeline-like main loop).

Build:
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_80 gemm_4_double_buffered.cu \
    -o gemm_4_double_buffered
Run:
  ./gemm_4_double_buffered [M] [N] [K] [device_id]
Example:
  ./gemm_4_double_buffered 2048 2048 2048 0

Notes:
  - Requires N % 4 == 0 and K % 4 == 0 (float4 LD/ST).
  - Next step: replace the gmem->smem path with cp.async (SM80+).
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

static_assert((kBM % kTM) == 0, "bad TM");
static_assert((kBN % kTN) == 0, "bad TN");

constexpr int kBlockX = kBN / kTN;  // 16
constexpr int kBlockY = kBM / kTM;  // 16

constexpr int kDefaultM = 2048;
constexpr int kDefaultN = 2048;
constexpr int kDefaultK = 2048;

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

__device__ __forceinline__ void SmemStoreATransposed(float (*s_a)[kBM + kPad],
                                                     int k4,
                                                     int m_row,
                                                     const float4& v) {
  s_a[k4 + 0][m_row] = v.x;
  s_a[k4 + 1][m_row] = v.y;
  s_a[k4 + 2][m_row] = v.z;
  s_a[k4 + 3][m_row] = v.w;
}

__global__ void GemmKernelDoubleBuffered(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ c,
                                         int m,
                                         int n,
                                         int k) {
  __shared__ float s_a[2][kBK][kBM + kPad];
  __shared__ float s_b[2][kBK][kBN + kPad];

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

  auto load_a_reg = [&](int k0) -> float4 {
    const int g_row = block_row + load_a_row;
    const int g_col = k0 + load_a_k4;
    if (g_row < m && (g_col + 3) < k) {
      return LoadFloat4(a + static_cast<size_t>(g_row) * static_cast<size_t>(k) +
                        static_cast<size_t>(g_col));
    }
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  };

  auto load_b_reg = [&](int k0) -> float4 {
    const int g_row = k0 + load_b_k;
    const int g_col = block_col + load_b_col4;
    if (g_row < k && (g_col + 3) < n) {
      return LoadFloat4(b + static_cast<size_t>(g_row) * static_cast<size_t>(n) +
                        static_cast<size_t>(g_col));
    }
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  };

  // Prefetch tile 0 -> smem buffer 0.
  {
    const float4 a_reg = load_a_reg(/*k0=*/0);
    const float4 b_reg = load_b_reg(/*k0=*/0);
    SmemStoreATransposed(s_a[0], load_a_k4, load_a_row, a_reg);
    StoreFloat4(&s_b[0][load_b_k][load_b_col4], b_reg);
  }
  __syncthreads();

  const int a_m_top = ty * (kTM / 2);
  const int a_m_bot = (kBM / 2) + ty * (kTM / 2);
  const int b_n_left = tx * (kTN / 2);
  const int b_n_right = (kBN / 2) + tx * (kTN / 2);

  // Main pipelined loop: t is the "next tile" we prefetch.
  for (int t = 1; t < num_tiles; ++t) {
    const int cur = (t - 1) & 1;
    const int next = t & 1;
    const int k0_next = t * kBK;

    // Prefetch next tile to registers.
    const float4 a_reg = load_a_reg(k0_next);
    const float4 b_reg = load_b_reg(k0_next);

    // Compute current tile from SMEM.
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

    // Write prefetched regs into the other SMEM buffer.
    SmemStoreATransposed(s_a[next], load_a_k4, load_a_row, a_reg);
    StoreFloat4(&s_b[next][load_b_k][load_b_col4], b_reg);

    __syncthreads();
  }

  // Compute last tile (already in SMEM).
  {
    const int last = (num_tiles - 1) & 1;
#pragma unroll
    for (int kk = 0; kk < kBK; ++kk) {
      const float4 a_top = LoadFloat4(&s_a[last][kk][a_m_top]);
      const float4 a_bot = LoadFloat4(&s_a[last][kk][a_m_bot]);
      const float4 b_l = LoadFloat4(&s_b[last][kk][b_n_left]);
      const float4 b_r = LoadFloat4(&s_b[last][kk][b_n_right]);

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
  }

  // Store: same 4-quadrant mapping as gemm_3.
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
    } else if (row_top < m) {
      for (int j = 0; j < (kTN / 2); ++j) {
        const int col0 = col_left + j;
        const int col1 = col_left + (kBN / 2) + j;
        if (col0 < n) {
          c[static_cast<size_t>(row_top) * static_cast<size_t>(n) +
            static_cast<size_t>(col0)] = r_c[i][j];
        }
        if (col1 < n) {
          c[static_cast<size_t>(row_top) * static_cast<size_t>(n) +
            static_cast<size_t>(col1)] = r_c[i][j + (kTN / 2)];
        }
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
    } else if (row_bot < m) {
      for (int j = 0; j < (kTN / 2); ++j) {
        const int col0 = col_left + j;
        const int col1 = col_left + (kBN / 2) + j;
        if (col0 < n) {
          c[static_cast<size_t>(row_bot) * static_cast<size_t>(n) +
            static_cast<size_t>(col0)] = r_c[i + (kTM / 2)][j];
        }
        if (col1 < n) {
          c[static_cast<size_t>(row_bot) * static_cast<size_t>(n) +
            static_cast<size_t>(col1)] =
              r_c[i + (kTM / 2)][j + (kTN / 2)];
        }
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
  const dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  GemmKernelDoubleBuffered<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
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
  std::printf("smem_pad:    %d\n", kPad);
  std::printf("time_ms:     %.3f\n", ms);
  std::printf("max_abs_err: %.6g\n", max_abs_err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return (max_abs_err <= 1e-2) ? 0 : 1;
}

