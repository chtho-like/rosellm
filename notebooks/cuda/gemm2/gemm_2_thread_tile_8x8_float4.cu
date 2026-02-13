/*
CUDA GEMM Interview Series v2 (C = A @ B)

File: gemm_2_thread_tile_8x8_float4.cu
Focus:
  - Block tiling in shared memory (reuse) +
    thread tiling in registers (compute density).
  - Each thread computes an 8x8 output tile in registers.
  - Vectorized global loads/stores using float4 (128-bit).

Why this matters:
  - In gemm_1, each thread computes 1 output -> not enough math per thread.
  - Here, we increase arithmetic intensity and reduce overhead.

Build:
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_80 gemm_2_thread_tile_8x8_float4.cu \
    -o gemm_2_thread_tile_8x8_float4
Run:
  ./gemm_2_thread_tile_8x8_float4 [M] [N] [K] [device_id]
Example:
  ./gemm_2_thread_tile_8x8_float4 512 512 512 0

Typical interviewer follow-ups:
  - Reduce shared-memory bank conflicts (padding / layout transforms).
  - Double-buffering to reduce __syncthreads frequency.
  - cp.async to overlap gmem->smem with compute (SM80+).
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

static_assert((kBM % kTM) == 0, "bad TM");
static_assert((kBN % kTN) == 0, "bad TN");

constexpr int kBlockX = kBN / kTN;  // 16
constexpr int kBlockY = kBM / kTM;  // 16

constexpr int kDefaultM = 512;
constexpr int kDefaultN = 512;
constexpr int kDefaultK = 512;

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

// Vectorized kernel: requires N % 4 == 0 and K % 4 == 0 for safe float4 LD/ST.
__global__ void GemmKernelThreadTile8x8Vec4(const float* __restrict__ a,
                                            const float* __restrict__ b,
                                            float* __restrict__ c,
                                            int m,
                                            int n,
                                            int k) {
  __shared__ float s_a[kBM][kBK];
  __shared__ float s_b[kBK][kBN];

  const int block_row = static_cast<int>(blockIdx.y) * kBM;
  const int block_col = static_cast<int>(blockIdx.x) * kBN;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  float acc[kTM][kTN];
#pragma unroll
  for (int i = 0; i < kTM; ++i) {
#pragma unroll
    for (int j = 0; j < kTN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  for (int k0 = 0; k0 < k; k0 += kBK) {
    // s_a is [kBM, kBK] and kBK=8 => 2 float4 per row, 128 rows => 256 float4.
    {
      const int a_row = tid / 2;
      const int a_col4 = (tid & 1) * 4;
      const int g_row = block_row + a_row;
      const int g_col = k0 + a_col4;
      float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (g_row < m && (g_col + 3) < k) {
        v = LoadFloat4(a + static_cast<size_t>(g_row) * static_cast<size_t>(k) +
                           static_cast<size_t>(g_col));
      }
      StoreFloat4(&s_a[a_row][a_col4], v);
    }

    // s_b is [kBK, kBN] and kBN=128 => 32 float4 per row, 8 rows => 256 float4.
    {
      const int b_row = tid / 32;
      const int b_col4 = (tid & 31) * 4;
      const int g_row = k0 + b_row;
      const int g_col = block_col + b_col4;
      float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (g_row < k && (g_col + 3) < n) {
        v = LoadFloat4(b + static_cast<size_t>(g_row) * static_cast<size_t>(n) +
                           static_cast<size_t>(g_col));
      }
      StoreFloat4(&s_b[b_row][b_col4], v);
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kBK; ++kk) {
      const float4 b0 = LoadFloat4(&s_b[kk][threadIdx.x * kTN + 0]);
      const float4 b1 = LoadFloat4(&s_b[kk][threadIdx.x * kTN + 4]);

#pragma unroll
      for (int i = 0; i < kTM; ++i) {
        const float a_val = s_a[threadIdx.y * kTM + i][kk];
        acc[i][0] = __fmaf_rn(a_val, b0.x, acc[i][0]);
        acc[i][1] = __fmaf_rn(a_val, b0.y, acc[i][1]);
        acc[i][2] = __fmaf_rn(a_val, b0.z, acc[i][2]);
        acc[i][3] = __fmaf_rn(a_val, b0.w, acc[i][3]);
        acc[i][4] = __fmaf_rn(a_val, b1.x, acc[i][4]);
        acc[i][5] = __fmaf_rn(a_val, b1.y, acc[i][5]);
        acc[i][6] = __fmaf_rn(a_val, b1.z, acc[i][6]);
        acc[i][7] = __fmaf_rn(a_val, b1.w, acc[i][7]);
      }
    }

    __syncthreads();
  }

  const int thread_row0 = block_row + threadIdx.y * kTM;
  const int thread_col0 = block_col + threadIdx.x * kTN;
  for (int i = 0; i < kTM; ++i) {
    const int row = thread_row0 + i;
    if (row >= m) {
      continue;
    }
    const int col = thread_col0;
    if (col >= n) {
      continue;
    }

    float* c_ptr = c + static_cast<size_t>(row) * static_cast<size_t>(n) +
                   static_cast<size_t>(col);
    if ((col + 7) < n) {
      StoreFloat4(c_ptr + 0,
                  make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]));
      StoreFloat4(c_ptr + 4,
                  make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]));
    } else {
      for (int j = 0; j < kTN; ++j) {
        if ((col + j) < n) {
          c_ptr[j] = acc[i][j];
        }
      }
    }
  }
}

// Scalar fallback (keeps the same tiling idea but avoids float4 requirements).
__global__ void GemmKernelThreadTile8x8Scalar(const float* __restrict__ a,
                                              const float* __restrict__ b,
                                              float* __restrict__ c,
                                              int m,
                                              int n,
                                              int k) {
  __shared__ float s_a[kBM][kBK];
  __shared__ float s_b[kBK][kBN];

  const int block_row = static_cast<int>(blockIdx.y) * kBM;
  const int block_col = static_cast<int>(blockIdx.x) * kBN;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  constexpr int kThreads = kBlockX * kBlockY;

  float acc[kTM][kTN];
#pragma unroll
  for (int i = 0; i < kTM; ++i) {
#pragma unroll
    for (int j = 0; j < kTN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  for (int k0 = 0; k0 < k; k0 += kBK) {
    for (int idx = tid; idx < (kBM * kBK); idx += kThreads) {
      const int i = idx / kBK;
      const int kk = idx - i * kBK;
      const int g_row = block_row + i;
      const int g_col = k0 + kk;
      s_a[i][kk] = (g_row < m && g_col < k)
                       ? a[static_cast<size_t>(g_row) * static_cast<size_t>(k) +
                           static_cast<size_t>(g_col)]
                       : 0.0f;
    }
    for (int idx = tid; idx < (kBK * kBN); idx += kThreads) {
      const int kk = idx / kBN;
      const int j = idx - kk * kBN;
      const int g_row = k0 + kk;
      const int g_col = block_col + j;
      s_b[kk][j] = (g_row < k && g_col < n)
                       ? b[static_cast<size_t>(g_row) * static_cast<size_t>(n) +
                           static_cast<size_t>(g_col)]
                       : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kBK; ++kk) {
#pragma unroll
      for (int i = 0; i < kTM; ++i) {
        const float a_val = s_a[threadIdx.y * kTM + i][kk];
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
          acc[i][j] = __fmaf_rn(a_val, s_b[kk][threadIdx.x * kTN + j],
                                acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  const int thread_row0 = block_row + threadIdx.y * kTM;
  const int thread_col0 = block_col + threadIdx.x * kTN;
  for (int i = 0; i < kTM; ++i) {
    const int row = thread_row0 + i;
    if (row >= m) {
      continue;
    }
    for (int j = 0; j < kTN; ++j) {
      const int col = thread_col0 + j;
      if (col >= n) {
        continue;
      }
      c[static_cast<size_t>(row) * static_cast<size_t>(n) +
        static_cast<size_t>(col)] = acc[i][j];
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
  const dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);

  const bool use_vec4 = ((n % 4) == 0) && ((k % 4) == 0);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  if (use_vec4) {
    GemmKernelThreadTile8x8Vec4<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  } else {
    GemmKernelThreadTile8x8Scalar<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  }
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
  std::printf("vec4_ldst:   %s\n", use_vec4 ? "yes" : "no");
  std::printf("time_ms:     %.3f\n", ms);
  std::printf("max_abs_err: %.6g\n", max_abs_err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return (max_abs_err <= 1e-3) ? 0 : 1;
}
