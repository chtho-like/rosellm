/*
CUDA Scan Interview Series (prefix sum)

File: scan_1_hierarchical_exclusive.cu
Focus:
  - Exclusive scan for large arrays using a 3-kernel hierarchical pattern:
    1) Scan each block (1024 elems) and write block sums.
    2) Scan the block sums (single block).
    3) Add block offsets to every element.

Interview tags:
  - Classic:     Very High
  - Importance:  High
  - Frequency:   High

Memorize (recommended):
  - YES. This 3-step pattern is the canonical "large scan" answer.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 scan_1_hierarchical_exclusive.cu \
    -o scan_1_hierarchical_exclusive
Run:
  ./scan_1_hierarchical_exclusive [n] [device_id]
Example:
  ./scan_1_hierarchical_exclusive 1048576 0

Limitations (kept small on purpose for interview memorization):
  - This demo requires num_blocks <= 1024 (so we can scan block_sums in one
    block). A fully general implementation recurses.
  - Production code often uses CUB: cub::DeviceScan.
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kThreads = 512;
constexpr int kElemsPerBlock = kThreads * 2;  // 1024
constexpr int kDefaultN = 1 << 20;           // 1,048,576

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

int CeilDivInt(int a, int b) {
  return (a + b - 1) / b;
}

void FillInput(std::vector<int>* x, int n) {
  x->resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    (*x)[static_cast<size_t>(i)] = (i % 13 == 0) ? 3 : 1;
  }
}

void CpuExclusiveScan(const std::vector<int>& x, std::vector<int>* y) {
  y->resize(x.size());
  int running = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    (*y)[i] = running;
    running += x[i];
  }
}

__device__ __forceinline__ int LoadOrZero(const int* in, int idx, int n) {
  return (idx < n) ? in[idx] : 0;
}

__global__ void BlockScanKernel(const int* __restrict__ in,
                                int* __restrict__ out,
                                int* __restrict__ block_sums,
                                int n) {
  __shared__ int smem[kElemsPerBlock];

  const int tid = threadIdx.x;
  const int base = static_cast<int>(blockIdx.x) * kElemsPerBlock;
  const int i0 = base + tid;
  const int i1 = base + tid + kThreads;

  smem[tid] = LoadOrZero(in, i0, n);
  smem[tid + kThreads] = LoadOrZero(in, i1, n);
  __syncthreads();

  for (int stride = 1; stride < kElemsPerBlock; stride <<= 1) {
    const int idx = (tid + 1) * stride * 2 - 1;
    if (idx < kElemsPerBlock) {
      smem[idx] += smem[idx - stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_sums[blockIdx.x] = smem[kElemsPerBlock - 1];
    smem[kElemsPerBlock - 1] = 0;
  }
  __syncthreads();

  for (int stride = kElemsPerBlock / 2; stride >= 1; stride >>= 1) {
    const int idx = (tid + 1) * stride * 2 - 1;
    if (idx < kElemsPerBlock) {
      const int t = smem[idx - stride];
      smem[idx - stride] = smem[idx];
      smem[idx] += t;
    }
    __syncthreads();
  }

  if (i0 < n) {
    out[i0] = smem[tid];
  }
  if (i1 < n) {
    out[i1] = smem[tid + kThreads];
  }
}

__global__ void ScanBlockSumsKernel(const int* __restrict__ in,
                                    int* __restrict__ out,
                                    int n) {
  __shared__ int smem[kElemsPerBlock];

  const int tid = threadIdx.x;
  const int i0 = tid;
  const int i1 = tid + kThreads;

  smem[i0] = (i0 < n) ? in[i0] : 0;
  smem[i1] = (i1 < n) ? in[i1] : 0;
  __syncthreads();

  for (int stride = 1; stride < kElemsPerBlock; stride <<= 1) {
    const int idx = (tid + 1) * stride * 2 - 1;
    if (idx < kElemsPerBlock) {
      smem[idx] += smem[idx - stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    smem[kElemsPerBlock - 1] = 0;
  }
  __syncthreads();

  for (int stride = kElemsPerBlock / 2; stride >= 1; stride >>= 1) {
    const int idx = (tid + 1) * stride * 2 - 1;
    if (idx < kElemsPerBlock) {
      const int t = smem[idx - stride];
      smem[idx - stride] = smem[idx];
      smem[idx] += t;
    }
    __syncthreads();
  }

  if (i0 < n) {
    out[i0] = smem[i0];
  }
  if (i1 < n) {
    out[i1] = smem[i1];
  }
}

__global__ void AddBlockOffsetsKernel(int* __restrict__ data,
                                      const int* __restrict__ block_offsets,
                                      int n) {
  const int base = static_cast<int>(blockIdx.x) * kElemsPerBlock;
  const int i0 = base + threadIdx.x;
  const int i1 = base + threadIdx.x + kThreads;
  const int offset = block_offsets[blockIdx.x];

  if (i0 < n) {
    data[i0] += offset;
  }
  if (i1 < n) {
    data[i1] += offset;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int n = (argc >= 2) ? ParseInt(argv[1], kDefaultN) : kDefaultN;
  const int device_id = (argc >= 3) ? ParseInt(argv[2], 0) : 0;

  if (n <= 0) {
    std::fprintf(stderr, "n must be > 0\n");
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(device_id));

  std::vector<int> h_in;
  FillInput(&h_in, n);
  std::vector<int> h_out_cpu;
  CpuExclusiveScan(h_in, &h_out_cpu);

  const int num_blocks = CeilDivInt(n, kElemsPerBlock);
  if (num_blocks > kElemsPerBlock) {
    std::fprintf(stderr,
                 "num_blocks=%d is too large for this demo (max 1024).\n",
                 num_blocks);
    return 2;
  }

  int* d_in = nullptr;
  int* d_out = nullptr;
  int* d_block_sums = nullptr;
  int* d_block_offsets = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(n) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_block_sums, static_cast<size_t>(num_blocks) *
                                         sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_block_offsets, static_cast<size_t>(num_blocks) *
                                            sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), static_cast<size_t>(n) * sizeof(int),
                        cudaMemcpyHostToDevice));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  BlockScanKernel<<<num_blocks, kThreads>>>(d_in, d_out, d_block_sums, n);
  CUDA_CHECK(cudaGetLastError());
  ScanBlockSumsKernel<<<1, kThreads>>>(d_block_sums, d_block_offsets,
                                       num_blocks);
  CUDA_CHECK(cudaGetLastError());
  AddBlockOffsetsKernel<<<num_blocks, kThreads>>>(d_out, d_block_offsets, n);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<int> h_out_gpu(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, static_cast<size_t>(n) *
                        sizeof(int), cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int i = 0; i < n; ++i) {
    if (h_out_gpu[static_cast<size_t>(i)] != h_out_cpu[static_cast<size_t>(i)]) {
      bad = 1;
      break;
    }
  }

  std::printf("n:         %d\n", n);
  std::printf("num_blocks: %d\n", num_blocks);
  std::printf("time_ms:    %.3f\n", ms);
  std::printf("ok:         %s\n", bad ? "no" : "yes");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_block_sums));
  CUDA_CHECK(cudaFree(d_block_offsets));

  return bad ? 1 : 0;
}

