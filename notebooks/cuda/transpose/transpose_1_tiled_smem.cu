/*
CUDA Transpose Interview Series

File: transpose_1_tiled_smem.cu
Focus:
  - Tile into shared memory:
    - coalesced reads from A
    - coalesced writes to B
  - This is the standard first optimization step for transpose.

Interview tags:
  - Classic:     High
  - Importance:  Medium
  - Frequency:   Medium

Memorize:
  - YES. Tiled transpose is a classic memory-coalescing exercise.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 transpose_1_tiled_smem.cu \
    -o transpose_1_tiled_smem
Run:
  ./transpose_1_tiled_smem [rows] [cols] [device_id]
Example:
  ./transpose_1_tiled_smem 1024 1024 0

Typical interviewer follow-up:
  - Avoid shared-memory bank conflicts with padding (tile[32][33]).
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kTile = 32;
constexpr int kBlockY = 8;
constexpr int kDefaultRows = 1024;
constexpr int kDefaultCols = 1024;

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

void FillInput(std::vector<float>* a, int rows, int cols) {
  a->resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      (*a)[static_cast<size_t>(r) * static_cast<size_t>(cols) +
           static_cast<size_t>(c)] = static_cast<float>((r * 131 + c * 17) % 97);
    }
  }
}

void CpuTranspose(const std::vector<float>& a,
                  std::vector<float>* b,
                  int rows,
                  int cols) {
  b->assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      (*b)[static_cast<size_t>(c) * static_cast<size_t>(rows) +
           static_cast<size_t>(r)] =
          a[static_cast<size_t>(r) * static_cast<size_t>(cols) +
            static_cast<size_t>(c)];
    }
  }
}

__global__ void TransposeKernelTiled(const float* __restrict__ a,
                                     float* __restrict__ b,
                                     int rows,
                                     int cols) {
  __shared__ float tile[kTile][kTile];

  const int x = static_cast<int>(blockIdx.x) * kTile + threadIdx.x;
  const int y = static_cast<int>(blockIdx.y) * kTile + threadIdx.y;

  for (int j = 0; j < kTile; j += kBlockY) {
    const int yy = y + j;
    if (x < cols && yy < rows) {
      tile[threadIdx.y + j][threadIdx.x] =
          a[static_cast<size_t>(yy) * static_cast<size_t>(cols) +
            static_cast<size_t>(x)];
    }
  }

  __syncthreads();

  const int x2 = static_cast<int>(blockIdx.y) * kTile + threadIdx.x;
  const int y2 = static_cast<int>(blockIdx.x) * kTile + threadIdx.y;

  for (int j = 0; j < kTile; j += kBlockY) {
    const int yy = y2 + j;
    if (x2 < rows && yy < cols) {
      b[static_cast<size_t>(yy) * static_cast<size_t>(rows) +
        static_cast<size_t>(x2)] =
          tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int rows = (argc >= 2) ? ParseInt(argv[1], kDefaultRows) : kDefaultRows;
  const int cols = (argc >= 3) ? ParseInt(argv[2], kDefaultCols) : kDefaultCols;
  const int device_id = (argc >= 4) ? ParseInt(argv[3], 0) : 0;

  if (rows <= 0 || cols <= 0) {
    std::fprintf(stderr, "rows and cols must be > 0\n");
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(device_id));

  std::vector<float> h_a;
  FillInput(&h_a, rows, cols);
  std::vector<float> h_b_cpu;
  CpuTranspose(h_a, &h_b_cpu, rows, cols);

  float* d_a = nullptr;
  float* d_b = nullptr;
  const size_t a_bytes =
      static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
  const size_t b_bytes = a_bytes;
  CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
  CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice));

  const dim3 block(kTile, kBlockY);
  const dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  TransposeKernelTiled<<<grid, block>>>(d_a, d_b, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<float> h_b_gpu(static_cast<size_t>(rows) *
                             static_cast<size_t>(cols));
  CUDA_CHECK(cudaMemcpy(h_b_gpu.data(), d_b, b_bytes, cudaMemcpyDeviceToHost));

  double max_abs_err = 0.0;
  for (size_t i = 0; i < h_b_gpu.size(); ++i) {
    const double diff = std::abs(static_cast<double>(h_b_gpu[i]) -
                                 static_cast<double>(h_b_cpu[i]));
    max_abs_err = std::max(max_abs_err, diff);
  }

  std::printf("rows:        %d\n", rows);
  std::printf("cols:        %d\n", cols);
  std::printf("time_ms:     %.3f\n", ms);
  std::printf("max_abs_err: %.6g\n", max_abs_err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));

  return (max_abs_err == 0.0) ? 0 : 1;
}

