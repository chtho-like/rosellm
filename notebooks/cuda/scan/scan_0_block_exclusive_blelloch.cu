/*
CUDA Scan Interview Series (prefix sum)

File: scan_0_block_exclusive_blelloch.cu
Focus:
  - Exclusive scan (prefix sum) within ONE block using Blelloch scan.
  - Each block scans up to 1024 elements (512 threads, 2 elems/thread).

Interview tags:
  - Classic:     Very High
  - Importance:  High (scan is a building block: compaction, radix sort, etc.)
  - Frequency:   High

Memorize (recommended):
  - YES. Blelloch scan is the canonical "whiteboard scan" algorithm.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 scan_0_block_exclusive_blelloch.cu \
    -o scan_0_block_exclusive_blelloch
Run:
  ./scan_0_block_exclusive_blelloch [n<=1024] [device_id]
Example:
  ./scan_0_block_exclusive_blelloch 1024 0

Notes:
  - This is a building block. For large arrays you need a hierarchical scan:
    scan_1_hierarchical_exclusive.cu.
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kThreads = 512;
constexpr int kElemsPerBlock = kThreads * 2;  // 1024
constexpr int kDefaultN = kElemsPerBlock;

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

void FillInput(std::vector<int>* x, int n) {
  x->resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    // Small ints make it easy to spot issues.
    (*x)[static_cast<size_t>(i)] = (i % 7 == 0) ? 2 : 1;
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

__global__ void BlockExclusiveScanKernel(const int* __restrict__ in,
                                         int* __restrict__ out,
                                         int n) {
  __shared__ int smem[kElemsPerBlock];

  const int tid = threadIdx.x;
  const int i0 = tid;
  const int i1 = tid + kThreads;

  smem[i0] = (i0 < n) ? in[i0] : 0;
  smem[i1] = (i1 < n) ? in[i1] : 0;
  __syncthreads();

  // Upsweep (reduce) phase.
  for (int stride = 1; stride < kElemsPerBlock; stride <<= 1) {
    const int idx = (tid + 1) * stride * 2 - 1;
    if (idx < kElemsPerBlock) {
      smem[idx] += smem[idx - stride];
    }
    __syncthreads();
  }

  // Set last element to 0 for exclusive scan.
  if (tid == 0) {
    smem[kElemsPerBlock - 1] = 0;
  }
  __syncthreads();

  // Downsweep phase.
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

}  // namespace

int main(int argc, char** argv) {
  const int n = (argc >= 2) ? ParseInt(argv[1], kDefaultN) : kDefaultN;
  const int device_id = (argc >= 3) ? ParseInt(argv[2], 0) : 0;

  if (n <= 0 || n > kElemsPerBlock) {
    std::fprintf(stderr, "n must be in (0, %d]\n", kElemsPerBlock);
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(device_id));

  std::vector<int> h_in;
  FillInput(&h_in, n);
  std::vector<int> h_out_cpu;
  CpuExclusiveScan(h_in, &h_out_cpu);

  int* d_in = nullptr;
  int* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(n) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), static_cast<size_t>(n) * sizeof(int),
                        cudaMemcpyHostToDevice));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  BlockExclusiveScanKernel<<<1, kThreads>>>(d_in, d_out, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<int> h_out_gpu(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out,
                        static_cast<size_t>(n) * sizeof(int),
                        cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int i = 0; i < n; ++i) {
    if (h_out_gpu[static_cast<size_t>(i)] != h_out_cpu[static_cast<size_t>(i)]) {
      bad = 1;
      break;
    }
  }

  std::printf("n:       %d\n", n);
  std::printf("time_ms: %.3f\n", ms);
  std::printf("ok:      %s\n", bad ? "no" : "yes");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  return bad ? 1 : 0;
}

