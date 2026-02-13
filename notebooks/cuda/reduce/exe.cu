#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>
#define CHECK(e) \
    do { \
        cudaError_t err = (e); \
        if(err != cudaSuccess) { \
            std::cout << __FILE__ << ":" << __LINE__ << " " \
                << cudaGetErrorString(err) << std::endl; \
            std::exit(1); \
        } \
    } while(false)
constexpr int kBlockSize = 256;
constexpr int kItemsPerThread = 2;
constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
__global__ void Reduce(const float* in, float* out, int n) {
    __shared__ float smem[kBlockSize];
    int tid = threadIdx.x;
    int idx = blockIdx.x * kItemsPerBlock + tid;
    int stride = gridDim.x * kItemsPerBlock;
    float s = 0;
    for(int i = idx; i < n; i += stride) {
        int i0 = idx;
        int i1 = idx + kBlockSize;
        if(i0 < n) s += in[i0];
        if(i1 < n) s += in[i1];
    }
    smem[tid] = s;
    __syncthreads();
    for(int i = kBlockSize / 2; i > 32; i >>= 1) {
        if(tid < i)
            smem[tid] += smem[tid+i];
        __syncthreads();
    }
    if(tid < 32) {
        float v = smem[tid];
        if(kBlockSize >= 64) v += smem[tid + 32];
        for(int i = 16; i > 0; i >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, i);
        }
        if(tid == 0) {
            out[blockIdx.x] = v;
        }
    }
}
float* ReduceAll(float* in, float* out, int n) {
    auto a = in;
    auto b = out;
    int rem = n;
    while(rem > 1) {
        auto blks = (rem + kItemsPerBlock - 1) / kItemsPerBlock;
        Reduce<<<blks, kBlockSize>>>(a, b, rem);
        CHECK(cudaGetLastError());
        rem = blks;
        std::swap(a, b);
    }
    return a;
}
int main() {
    const int n = 100;
    std::vector<float> h_in(n);
    double cpu_sum = 0.0;
    for(int i = 0; i < n; i++) {
        h_in[i] = rand();
        cpu_sum += h_in[i];
    }
    float* in = nullptr;
    float* out = nullptr;
    CHECK(cudaMalloc(&in, n * sizeof(float)));
    CHECK(cudaMalloc(&out, n * sizeof(float)));
    CHECK(cudaMemcpy(in, h_in.data(), n * sizeof(float), 
        cudaMemcpyHostToDevice));
    float* c = ReduceAll(in, out, n);
    CHECK(cudaDeviceSynchronize());
    float gpu_sum_f = 0.0f;
    CHECK(cudaMemcpy(&gpu_sum_f, c, sizeof(gpu_sum_f),
        cudaMemcpyDeviceToHost));
    double gpu_sum = static_cast<double>(gpu_sum_f);
    double abs_err = std::abs(gpu_sum - cpu_sum);
    double rel_err = abs_err;
    if(cpu_sum != 0.0) rel_err /= std::abs(cpu_sum);
    std::cout << "cpu_sum: " << cpu_sum << std::endl;
    std::cout << "gpu_sum: " << gpu_sum << std::endl;
    std::cout << "abs_err: " << abs_err << std::endl;
    std::cout << "rel_err: " << rel_err << std::endl;
    CHECK(cudaFree(in));
    CHECK(cudaFree(out));
}
