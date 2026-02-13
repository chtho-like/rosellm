/*
CUDA Sort Interview Series (bitonic sort, key-value)

File: sort_1_bitonic_key_value.cu
Focus:
  - Bitonic sort of (key, value) pairs in shared memory (up to 1024).
  - This is closer to real use: you sort keys and carry payload/indices.

Interview tags:
  - Classic:     High
  - Importance:  Medium
  - Frequency:   Medium

Memorize:
  - Recommended if you already know sort_0. Key-value makes it "real".

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 sort_1_bitonic_key_value.cu \
    -o sort_1_bitonic_key_value
Run:
  ./sort_1_bitonic_key_value [n<=1024,pow2] [device_id]
Example:
  ./sort_1_bitonic_key_value 1024 0
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
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

void FillInput(std::vector<int>* keys, std::vector<int>* vals, int n) {
  keys->resize(static_cast<size_t>(n));
  vals->resize(static_cast<size_t>(n));
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-1000, 1000);
  for (int i = 0; i < n; ++i) {
    (*keys)[static_cast<size_t>(i)] = dist(rng);
    (*vals)[static_cast<size_t>(i)] = i;
  }
}

struct Pair {
  int key;
  int val;
};

__device__ __forceinline__ void CompareSwap(Pair& a, Pair& b, bool ascending) {
  // Define a total order to make tests deterministic:
  //   (key, val) ascending.
  const bool swap =
      ascending
          ? ((a.key > b.key) || ((a.key == b.key) && (a.val > b.val)))
          : ((a.key < b.key) || ((a.key == b.key) && (a.val < b.val)));
  if (swap) {
    const Pair t = a;
    a = b;
    b = t;
  }
}

__global__ void BitonicSortPairsKernel(const int* __restrict__ in_keys,
                                       const int* __restrict__ in_vals,
                                       int* __restrict__ out_keys,
                                       int* __restrict__ out_vals,
                                       int n) {
  __shared__ Pair data[kMaxN];
  const int tid = threadIdx.x;
  if (tid < n) {
    data[tid] = Pair{in_keys[tid], in_vals[tid]};
  } else {
    data[tid] = Pair{INT_MAX, -1};
  }
  __syncthreads();

  for (int k = 2; k <= kMaxN; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int ixj = tid ^ j;
      if (ixj > tid) {
        const bool ascending = ((tid & k) == 0);
        Pair a = data[tid];
        Pair b = data[ixj];
        CompareSwap(a, b, ascending);
        data[tid] = a;
        data[ixj] = b;
      }
      __syncthreads();
    }
  }

  if (tid < n) {
    out_keys[tid] = data[tid].key;
    out_vals[tid] = data[tid].val;
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

  std::vector<int> h_keys;
  std::vector<int> h_vals;
  FillInput(&h_keys, &h_vals, n);

  std::vector<std::pair<int, int>> pairs(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    pairs[static_cast<size_t>(i)] =
        {h_keys[static_cast<size_t>(i)], h_vals[static_cast<size_t>(i)]};
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const auto& a, const auto& b) {
              if (a.first != b.first) {
                return a.first < b.first;
              }
              return a.second < b.second;
            });

  int* d_keys = nullptr;
  int* d_vals = nullptr;
  int* d_out_keys = nullptr;
  int* d_out_vals = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, static_cast<size_t>(n) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_vals, static_cast<size_t>(n) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out_keys, static_cast<size_t>(n) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out_vals, static_cast<size_t>(n) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), static_cast<size_t>(n) *
                        sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vals, h_vals.data(), static_cast<size_t>(n) *
                        sizeof(int), cudaMemcpyHostToDevice));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  BitonicSortPairsKernel<<<1, kBlockSize>>>(d_keys, d_vals, d_out_keys,
                                            d_out_vals, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<int> h_out_keys(static_cast<size_t>(n));
  std::vector<int> h_out_vals(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(h_out_keys.data(), d_out_keys, static_cast<size_t>(n) *
                        sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_out_vals.data(), d_out_vals, static_cast<size_t>(n) *
                        sizeof(int), cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int i = 0; i < n; ++i) {
    const auto& p = pairs[static_cast<size_t>(i)];
    if (h_out_keys[static_cast<size_t>(i)] != p.first ||
        h_out_vals[static_cast<size_t>(i)] != p.second) {
      bad = 1;
      break;
    }
  }

  std::printf("n:       %d\n", n);
  std::printf("time_ms: %.3f\n", ms);
  std::printf("ok:      %s\n", bad ? "no" : "yes");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vals));
  CUDA_CHECK(cudaFree(d_out_keys));
  CUDA_CHECK(cudaFree(d_out_vals));

  return bad ? 1 : 0;
}
