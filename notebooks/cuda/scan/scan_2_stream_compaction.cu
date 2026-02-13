/*
CUDA Scan Interview Series (stream compaction)

File: scan_2_stream_compaction.cu
Focus:
  - Stream compaction: keep elements that satisfy a predicate.
  - Classic pattern:
    1) flags[i] = predicate(in[i]) ? 1 : 0
    2) pos = exclusive_scan(flags)
    3) if flags[i] == 1: out[pos[i]] = in[i]

Interview tags:
  - Classic:     Very High
  - Importance:  High (used in graph processing, filtering, radix sort)
  - Frequency:   High

Memorize (recommended):
  - YES. "compaction via scan" is a top-tier interview building block.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 scan_2_stream_compaction.cu \
    -o scan_2_stream_compaction
Run:
  ./scan_2_stream_compaction [n] [device_id]
Example:
  ./scan_2_stream_compaction 1048576 0

Limitations:
  - Uses the same simplified hierarchical scan as scan_1, so requires
    num_blocks <= 1024.
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kThreads = 512;
constexpr int kElemsPerBlock = kThreads * 2;  // 1024
constexpr int kDefaultN = 1 << 20;
constexpr int kIoBlock = 256;

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
    // Keep about 1/4 of the elements.
    (*x)[static_cast<size_t>(i)] = (i % 4 == 0) ? (i + 1) : 0;
  }
}

void CpuCompact(const std::vector<int>& in, std::vector<int>* out) {
  out->clear();
  out->reserve(in.size());
  for (int v : in) {
    if (v != 0) {
      out->push_back(v);
    }
  }
}

__global__ void FlagKernel(const int* __restrict__ in,
                           int* __restrict__ flags,
                           int n) {
  const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid < n) {
    flags[tid] = (in[tid] != 0) ? 1 : 0;
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

  if (i0 < n) out[i0] = smem[tid];
  if (i1 < n) out[i1] = smem[tid + kThreads];
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

  if (i0 < n) out[i0] = smem[i0];
  if (i1 < n) out[i1] = smem[i1];
}

__global__ void AddBlockOffsetsKernel(int* __restrict__ data,
                                      const int* __restrict__ block_offsets,
                                      int n) {
  const int base = static_cast<int>(blockIdx.x) * kElemsPerBlock;
  const int i0 = base + threadIdx.x;
  const int i1 = base + threadIdx.x + kThreads;
  const int offset = block_offsets[blockIdx.x];

  if (i0 < n) data[i0] += offset;
  if (i1 < n) data[i1] += offset;
}

__global__ void ScatterKernel(const int* __restrict__ in,
                              const int* __restrict__ flags,
                              const int* __restrict__ pos,
                              int* __restrict__ out,
                              int n) {
  const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid < n && flags[tid] != 0) {
    out[pos[tid]] = in[tid];
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
  CpuCompact(h_in, &h_out_cpu);

  const int num_blocks_scan = CeilDivInt(n, kElemsPerBlock);
  if (num_blocks_scan > kElemsPerBlock) {
    std::fprintf(stderr,
                 "num_blocks=%d is too large for this demo (max 1024).\n",
                 num_blocks_scan);
    return 2;
  }

  int* d_in = nullptr;
  int* d_flags = nullptr;
  int* d_pos = nullptr;
  int* d_block_sums = nullptr;
  int* d_block_offsets = nullptr;
  int* d_out = nullptr;

  CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(n) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_flags, static_cast<size_t>(n) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_pos, static_cast<size_t>(n) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_block_sums,
                        static_cast<size_t>(num_blocks_scan) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_block_offsets,
                        static_cast<size_t>(num_blocks_scan) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), static_cast<size_t>(n) * sizeof(int),
                        cudaMemcpyHostToDevice));

  const dim3 io_grid(static_cast<unsigned int>(CeilDivInt(n, kIoBlock)), 1, 1);
  const dim3 io_block(kIoBlock, 1, 1);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  FlagKernel<<<io_grid, io_block>>>(d_in, d_flags, n);
  CUDA_CHECK(cudaGetLastError());

  BlockScanKernel<<<num_blocks_scan, kThreads>>>(d_flags, d_pos, d_block_sums,
                                                 n);
  CUDA_CHECK(cudaGetLastError());
  ScanBlockSumsKernel<<<1, kThreads>>>(d_block_sums, d_block_offsets,
                                       num_blocks_scan);
  CUDA_CHECK(cudaGetLastError());
  AddBlockOffsetsKernel<<<num_blocks_scan, kThreads>>>(d_pos, d_block_offsets,
                                                       n);
  CUDA_CHECK(cudaGetLastError());

  ScatterKernel<<<io_grid, io_block>>>(d_in, d_flags, d_pos, d_out, n);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  // Compute out_size = pos[n-1] + flags[n-1].
  int last_pos = 0;
  int last_flag = 0;
  CUDA_CHECK(cudaMemcpy(&last_pos, d_pos + (n - 1), sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&last_flag, d_flags + (n - 1), sizeof(int),
                        cudaMemcpyDeviceToHost));
  const int out_size = last_pos + last_flag;

  std::vector<int> h_out_gpu(static_cast<size_t>(out_size));
  CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out,
                        static_cast<size_t>(out_size) * sizeof(int),
                        cudaMemcpyDeviceToHost));

  int bad = 0;
  if (static_cast<int>(h_out_cpu.size()) != out_size) {
    bad = 1;
  } else {
    for (int i = 0; i < out_size; ++i) {
      if (h_out_gpu[static_cast<size_t>(i)] !=
          h_out_cpu[static_cast<size_t>(i)]) {
        bad = 1;
        break;
      }
    }
  }

  std::printf("n:         %d\n", n);
  std::printf("out_size:   %d\n", out_size);
  std::printf("time_ms:    %.3f\n", ms);
  std::printf("ok:         %s\n", bad ? "no" : "yes");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_flags));
  CUDA_CHECK(cudaFree(d_pos));
  CUDA_CHECK(cudaFree(d_block_sums));
  CUDA_CHECK(cudaFree(d_block_offsets));
  CUDA_CHECK(cudaFree(d_out));

  return bad ? 1 : 0;
}

