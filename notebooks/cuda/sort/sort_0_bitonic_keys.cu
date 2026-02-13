/*
CUDA Sort Interview Series (bitonic sort)

File: sort_0_bitonic_keys.cu
Focus:
  - Bitonic sort in shared memory for up to 1024 keys.
  - This is a classic CUDA interview problem for demonstrating
    shared memory + synchronization + compare-and-swap networks.

Interview tags:
  - Classic:     High
  - Importance:  Medium (sorting itself is less common in LLM inference)
  - Frequency:   Medium

Memorize:
  - YES (if you expect general GPU interviews). Bitonic is very common.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 sort_0_bitonic_keys.cu \
    -o sort_0_bitonic_keys
Run:
  ./sort_0_bitonic_keys [n<=1024,pow2] [device_id]
Example:
  ./sort_0_bitonic_keys 1024 0
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr int kMaxN = 1024;
constexpr int kBlockSize = kMaxN;
constexpr int kDefaultN = kMaxN;

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

bool IsPowerOfTwo(int x) {
  return (x > 0) && ((x & (x - 1)) == 0);
}

void FillInput(std::vector<int>* x, int n) {
  x->resize(static_cast<size_t>(n));
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-1000, 1000);
  for (int i = 0; i < n; ++i) {
    (*x)[static_cast<size_t>(i)] = dist(rng);
  }
}

__device__ __forceinline__ void CompareSwap(int& a, int& b, bool ascending) {
  const bool swap = ascending ? (a > b) : (a < b);
  if (swap) {
    const int t = a;
    a = b;
    b = t;
  }
}

__global__ void BitonicSortKernel(const int* __restrict__ in,
                                  int* __restrict__ out,
                                  int n) {
  __shared__ int data[kMaxN];
  const int tid = threadIdx.x;
  data[tid] = (tid < n) ? in[tid] : INT_MAX;
  __syncthreads();

  // Bitonic sort network over 1024 elements.
  for (int k = 2; k <= kMaxN; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int ixj = tid ^ j;
      if (ixj > tid) {
        const bool ascending = ((tid & k) == 0);
        int a = data[tid];
        int b = data[ixj];
        CompareSwap(a, b, ascending);
        data[tid] = a;
        data[ixj] = b;
      }
      __syncthreads();
    }
  }

  if (tid < n) {
    out[tid] = data[tid];
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int n = (argc >= 2) ? ParseInt(argv[1], kDefaultN) : kDefaultN;
  const int device_id = (argc >= 3) ? ParseInt(argv[2], 0) : 0;

  if (n <= 0 || n > kMaxN || !IsPowerOfTwo(n)) {
    std::fprintf(stderr, "n must be power-of-two in (0, %d]\n", kMaxN);
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(device_id));

  std::vector<int> h_in;
  FillInput(&h_in, n);
  std::vector<int> h_sorted_cpu = h_in;
  std::sort(h_sorted_cpu.begin(), h_sorted_cpu.end());

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
  BitonicSortKernel<<<1, kBlockSize>>>(d_in, d_out, n);
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
    if (h_out_gpu[static_cast<size_t>(i)] !=
        h_sorted_cpu[static_cast<size_t>(i)]) {
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

