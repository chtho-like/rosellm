#include <cuda_runtime.h>

#include <iostream>
#include <cstdio>
#include <vector>


namespace {

constexpr int kBlockSize = 256;
constexpr int kItemsPerThread = 1;
constexpr size_t kDefaultNumElements = 1u << 24;

#define CUDA_CHECK(expr) \
  do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
  } while (false)

__global__ void ReduceKernel(const float* __restrict__ in,
                             float* __restrict__ out,
                             size_t n) {
    __shared__ float sdata[kBlockSize];
    const unsigned int tid = threadIdx.x;
    const size_t idx = static_cast<size_t>(blockIdx.x) * 
        static_cast<size_t>(kBlockSize) + static_cast<size_t>(tid);
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();
    for (unsigned int stride = 1; stride < kBlockSize; stride <<= 1) {
        if ((tid % (2 * stride)) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

__host__ __device__ constexpr size_t CeilDivSizeT(size_t a, size_t b) {
    return (a + b - 1) / b;
}

float* GpuReduceSumInPlace(float* d_in, float* d_tmp, size_t n) {
    float* in = d_in;
    float* out = d_tmp;
    size_t remaining = n;
    const size_t elements_per_block = 
        static_cast<size_t>(kBlockSize) * static_cast<size_t>(kItemsPerThread);
    while (remaining > 1) {
        const int blocks = 
            static_cast<int>(CeilDivSizeT(remaining, elements_per_block));
        ReduceKernel<<<blocks, kBlockSize>>>(in, out, remaining);
        CUDA_CHECK(cudaGetLastError());
        remaining = static_cast<size_t>(blocks);
        std::swap(in, out);
    }
    return in;
}

double CpuSum(const std::vector<float>& x) {
    double sum = 0.0;
    for (float v: x) {
        sum += static_cast<double>(v);
    }
    return sum;
}

}  // namespace

/*
warp size = 32
max threads per block = 1024
max registers per thread = 255 (typically) (ps: every register is of 32-bit)
max static shared memory per block = 48KB (12k * float32)

cc: compute capability
* ws: max warps per sm (=ths/)
* ths: max threads per sm 
* bs: max blocks per sm
* regs: max registers per sm
* smem/sm: max shared memory per sm
* smem/blk: max shared memory per block
sm: number of sm
comb/sm: combined L1/texture/shared capacity (const for L1 min: ~28KB)

GPU    cc ws  ths bs regs comb/sm smem/sm smem/blk  sm   memory         bw   L2 NVLnk   FP32 BF16
4070  8.9 48 1536 24  64K   128KB  100 KB    99 KB  46  12GB GDDR6X  504G/s  36M   x     29T   x
A100  8.0 64 2048 32  64K   192KB  164 KB   163 KB 108  80GB  HBM2e    2T/s  40M  600G 19.5T 312T
H100  9.0 64 2048 32  64K   256KB  228 KB   227 KB 132  80GB  HBM3  3.35T/s  50M  900G   67T 989T
H200  9.0 64 2048 32  64K   256KB  228 KB   227 KB 132 141GB  HBM3e  4.8T/s  50M  900G   67T 989T
H800  9.0 64 2048 32  64K   256KB  228 KB   227 KB 144  96GB  HBM3     ?     60M  400G   51T   ?
H20   9.0 64 2048 32  64K   256KB  228 KB   227 KB  ?   96GB  HBM3   4.0T/s  60M  900G   44T   ?
B200 10.0 64 2048 32  64K   256KB  228 KB   227 KB  ?  180GB   ?       8T/s 126M  1.8T   75T 4.5P
*/

int main() {
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    int maxWarpsPerSM = p.maxThreadsPerMultiProcessor / p.warpSize;
    int maxBlocksPerSM = p.maxBlocksPerMultiProcessor;
    // 1536
    std::cout << "maxThreadsPerMultiProcessor: " << p.maxThreadsPerMultiProcessor << std::endl;
    // 48
    std::cout << "maxWarpsPerSM: " << maxWarpsPerSM << std::endl;
    // 24
    std::cout << "maxBlocksPerSM: " << maxBlocksPerSM << std::endl;
    int tsm = 0, ws = 0;
    cudaDeviceGetAttribute(&tsm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    cudaDeviceGetAttribute(&ws, cudaDevAttrWarpSize, 0);
    std::cout << "maxThreadsPerMultiProcessor: " << tsm << std::endl;
    std::cout << "warpSize: " << ws << std::endl;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(nullptr, nullptr, 256);
    
    const size_t num_elements = kDefaultNumElements;
    const int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));
    std::vector<float> h_in(num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        const int v = static_cast<int>(i % 3) - 1;
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
    float* d_result = GpuReduceSumInPlace(d_in, d_tmp, num_elements);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpu_sum_f = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_sum_f, d_result, sizeof(float),
        cudaMemcpyDeviceToHost));
    std::cout << "cpu sum: " << cpu_sum << std::endl;
    std::cout << "gpu sum: " << gpu_sum_f << std::endl;
}