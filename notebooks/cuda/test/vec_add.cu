#include <cuda_runtime.h>
#include <cstdio>

__global__ void vec_add(const float* a,
                        const float* b,
                        float* c,
                        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1 << 20;
    float *da, *db, *dc;
    cudaMalloc(&da, n * sizeof(float));
    cudaMalloc(&db, n * sizeof(float));
    cudaMalloc(&dc, n * sizeof(float));
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    vec_add<<<grid, block>>>(da, db, dc, n);
    cudaDeviceSynchronize();
    cudaFree(da); cudaFree(db); cudaFree(dc);
    printf("done\n");
}