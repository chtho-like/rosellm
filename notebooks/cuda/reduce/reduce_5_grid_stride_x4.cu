/*
CUDA Reduction Interview Series (sum reduction over a float array)

File: reduce_5_grid_stride_x4.cu
Focus:
  - Each thread loads more elements (x4) and accumulates in registers.
  - Use a grid-stride loop so a "small" grid can cover a very large input.

Typical interviewer follow-up after reduce_4:
  - "Now reduce kernel launch overhead and global reads."
  - "Show the grid-stride loop pattern."
  - "Optionally, cap the number of blocks to ~SMs * k."

Build:
  nvcc -O3 -std=c++17 -lineinfo reduce_5_grid_stride_x4.cu \
    -o reduce_5_grid_stride_x4
Run:
  ./reduce_5_grid_stride_x4 [num_elements] [device_id]
Example:
  ./reduce_5_grid_stride_x4 16777216 0
Notes:
  - The first run can include CUDA driver JIT compilation; run twice to
    compare timings.
  - References (searchable): NVIDIA "Optimizing Parallel Reduction in CUDA",
    CUDA Samples "reduction".
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kItemsPerThread = 4;
constexpr size_t kDefaultNumElements = 1u << 24;
constexpr unsigned int kFullWarpMask = 0xFFFF'FFFFu;
constexpr int kBlocksPerSm = 4;

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t err__ = (expr);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err__));                                \
      std::exit(1);                                                           \
    }                                                                         \
  } while (false)

__host__ __device__ constexpr size_t CeilDivSizeT(size_t a, size_t b) {
  return (a + b - 1) / b;
}

size_t ParseSizeT(const char* s, size_t default_value) {
  if (s == nullptr) {
    return default_value;
  }
  char* end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s) {
    return default_value;
  }
  return static_cast<size_t>(v);
}

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

double CpuSum(const std::vector<float>& x) {
  double sum = 0.0;
  for (float v : x) {
    sum += static_cast<double>(v);
  }
  return sum;
}

int GetSmCount() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  int sm_count = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                    device));
  return sm_count;
}

int ComputeNumBlocks(size_t n) {
  const size_t elements_per_block =
      static_cast<size_t>(kBlockSize) * static_cast<size_t>(kItemsPerThread);
  const int full_blocks = static_cast<int>(CeilDivSizeT(n, elements_per_block));
  const int max_blocks = GetSmCount() * kBlocksPerSm;
  return std::min(full_blocks, max_blocks);
}

__device__ __forceinline__ float WarpReduceSum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(kFullWarpMask, v, offset);
  }
  return v;
}

// Step 5 kernel:
// - Grid-stride loop: base += gridDim * blockDim * kItemsPerThread.
// - More work per thread (x4 loads) amortizes overheads.
__global__ void ReduceKernel(const float* __restrict__ in,
                             float* __restrict__ out,
                             size_t n) {
  __shared__ float sdata[kBlockSize];

  const unsigned int tid = threadIdx.x;
  const size_t block_items = static_cast<size_t>(kBlockSize) *
                             static_cast<size_t>(kItemsPerThread);
  const size_t block_base =
      static_cast<size_t>(blockIdx.x) * block_items + tid;
  const size_t grid_stride = static_cast<size_t>(gridDim.x) * block_items;

  float sum = 0.0f;
  for (size_t base = block_base; base < n; base += grid_stride) {
    const size_t idx0 = base;
    const size_t idx1 = base + static_cast<size_t>(kBlockSize);
    const size_t idx2 = idx1 + static_cast<size_t>(kBlockSize);
    const size_t idx3 = idx2 + static_cast<size_t>(kBlockSize);

    if (idx0 < n) {
      sum += in[idx0];
    }
    if (idx1 < n) {
      sum += in[idx1];
    }
    if (idx2 < n) {
      sum += in[idx2];
    }
    if (idx3 < n) {
      sum += in[idx3];
    }
  }

  sdata[tid] = sum;
  __syncthreads();

  for (unsigned int stride = kBlockSize / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  if (tid < 32) {
    float v = sdata[tid];
    if (kBlockSize >= 64) {
      v += sdata[tid + 32];
    }
    v = WarpReduceSum(v);
    if (tid == 0) {
      out[blockIdx.x] = v;
    }
  }
}

float* GpuReduceSumInplace(float* d_in, float* d_tmp, size_t n) {
  float* in = d_in;
  float* out = d_tmp;

  size_t remaining = n;
  while (remaining > 1) {
    const int blocks = ComputeNumBlocks(remaining);
    ReduceKernel<<<blocks, kBlockSize>>>(in, out, remaining);
    CUDA_CHECK(cudaGetLastError());
    remaining = static_cast<size_t>(blocks);
    std::swap(in, out);
  }

  return in;
}

}  // namespace

int main(int argc, char** argv) {
  const size_t num_elements =
      (argc >= 2) ? ParseSizeT(argv[1], kDefaultNumElements)
                  : kDefaultNumElements;
  const int device_id = (argc >= 3) ? ParseInt(argv[2], 0) : 0;

  if (num_elements == 0) {
    std::fprintf(stderr, "num_elements must be > 0\n");
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(device_id));

  std::vector<float> h_in(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    const int v = static_cast<int>(i % 3) - 1;  // -1, 0, 1 (exact in float)
    h_in[i] = static_cast<float>(v);
  }

  const double cpu_sum = CpuSum(h_in);

  float* d_in = nullptr;
  float* d_tmp = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, num_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tmp, num_elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), num_elements * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  float* d_result = GpuReduceSumInplace(d_in, d_tmp, num_elements);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  float gpu_sum_f = 0.0f;
  CUDA_CHECK(cudaMemcpy(&gpu_sum_f, d_result, sizeof(float),
                        cudaMemcpyDeviceToHost));
  const double gpu_sum = static_cast<double>(gpu_sum_f);

  const double abs_err = std::abs(gpu_sum - cpu_sum);
  const double rel_err =
      (cpu_sum == 0.0) ? abs_err : (abs_err / std::abs(cpu_sum));

  std::printf("num_elements: %zu\n", num_elements);
  std::printf("cpu_sum:      %.0f\n", cpu_sum);
  std::printf("gpu_sum:      %.0f\n", gpu_sum);
  std::printf("abs_err:      %.6g\n", abs_err);
  std::printf("rel_err:      %.6g\n", rel_err);
  std::printf("time_ms:      %.3f\n", ms);
  std::printf("blocks_used:  %d\n", ComputeNumBlocks(num_elements));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_tmp));

  return (rel_err <= 1e-6) ? 0 : 1;
}
