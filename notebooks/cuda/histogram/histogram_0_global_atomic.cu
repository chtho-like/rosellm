/*
CUDA Histogram Interview Series

File: histogram_0_global_atomic.cu
Focus:
  - Baseline histogram with global atomics.
  - Input values are uint8 in [0, 255], bins = 256.

Interview tags:
  - Classic:     Medium
  - Importance:  Medium (tests atomics + memory patterns)
  - Frequency:   Medium

Memorize:
  - Baseline only. The common follow-up is shared-memory privatization.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 histogram_0_global_atomic.cu \
    -o histogram_0_global_atomic
Run:
  ./histogram_0_global_atomic [n] [device_id]
Example:
  ./histogram_0_global_atomic 1048576 0

Typical interviewer follow-up:
  - Privatize histogram per block in shared memory, then merge to global.
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kBins = 256;
constexpr int kBlockSize = 256;
constexpr int kDefaultN = 1 << 20;

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

void FillInput(std::vector<std::uint8_t>* x, int n) {
  x->resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    (*x)[static_cast<size_t>(i)] =
        static_cast<std::uint8_t>((i * 131 + 7) & 0xFF);
  }
}

void CpuHistogram(const std::vector<std::uint8_t>& x,
                  std::vector<unsigned int>* bins) {
  bins->assign(kBins, 0u);
  for (std::uint8_t v : x) {
    (*bins)[static_cast<size_t>(v)] += 1u;
  }
}

__global__ void HistogramKernelGlobalAtomic(const std::uint8_t* __restrict__ x,
                                            unsigned int* __restrict__ bins,
                                            int n) {
  const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride = static_cast<int>(gridDim.x) * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    const std::uint8_t v = x[i];
    atomicAdd(&bins[static_cast<int>(v)], 1u);
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

  std::vector<std::uint8_t> h_x;
  FillInput(&h_x, n);
  std::vector<unsigned int> h_bins_cpu;
  CpuHistogram(h_x, &h_bins_cpu);

  std::uint8_t* d_x = nullptr;
  unsigned int* d_bins = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_bins, static_cast<size_t>(kBins) *
                                    sizeof(unsigned int)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), static_cast<size_t>(n) *
                        sizeof(std::uint8_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_bins, 0, static_cast<size_t>(kBins) *
                                sizeof(unsigned int)));

  const int blocks = 256;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  HistogramKernelGlobalAtomic<<<blocks, kBlockSize>>>(d_x, d_bins, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<unsigned int> h_bins_gpu(kBins);
  CUDA_CHECK(cudaMemcpy(h_bins_gpu.data(), d_bins,
                        static_cast<size_t>(kBins) * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int i = 0; i < kBins; ++i) {
    if (h_bins_gpu[static_cast<size_t>(i)] !=
        h_bins_cpu[static_cast<size_t>(i)]) {
      bad = 1;
      break;
    }
  }

  std::printf("n:       %d\n", n);
  std::printf("time_ms: %.3f\n", ms);
  std::printf("ok:      %s\n", bad ? "no" : "yes");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_bins));

  return bad ? 1 : 0;
}

