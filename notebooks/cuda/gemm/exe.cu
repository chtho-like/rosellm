#include <cassert>

__global__ void GEMMNaive(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M,
                          int N,
                          int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  float acc = 0.0f;
  for (int i = 0; i < K; ++i) {
    acc += A[row * K + i] * B[i * N + col];
  }
  C[row * N + col] = acc;
}

// Requires: blockDim.y = BM, blockDim.x = BN
// BM == BN == BK
template<int BM, int BK, int BN>
__global__ void GEMMBlockTile(float* A,
                              float* B,
                              float* C,
                              int M,
                              int N,
                              int K) {
    __shared__ float As[BM][BK], Bs[BK][BN];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;
    float acc = 0.0f;
    for(int k = 0; k < K; k += BK) {
        int a_col = k + tx;
        int b_row = k + ty;
        As[ty][tx] = (row >= M || a_col >= K) ? 0.0f : A[row*K + a_col];
        Bs[ty][tx] = (b_row >= K || col >= N) ? 0.0f : B[b_row*N + col];
        __syncthreads();
        #pragma unroll
        for(int i = 0; i < BK; i++)
            acc += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    if(row < M && col < N) C[row*N + col] = acc;
}

// Requires: BM = blockDim.y * TM, BN = blockDim.x * TN
template<int BM, int BK, int BN, int TM, int TN>
__global__ void GEMMThreadTile(float* A,
                               float* B,
                               float* C,
                               int M,
                               int N, 
                               int K) {
    static_assert(BM % TM == 0);
    static_assert(BN % TN == 0);
    __shared__ float As[BM][BK], Bs[BK][BN];
    int tx = threadIdx.x, ty = threadIdx.y;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int thread_row0 = block_row + ty * TM;
    int thread_col0 = block_col + tx * TN;
    float acc[TM][TN] = {0.0f};
    for(int k = 0; k < K; k += BK) {
        int tid = tx + ty * blockDim.x;
        int ts = blockDim.x * blockDim.y;
        for(int i = tid; i < BM*BK; i += ts) {
            int row = i / BK;
            int col = i - row*BK;
            int a_row = block_row + row;
            int a_col = k + col;
            As[row][col] = (a_row < M && a_col < K) ? A[a_row*K + a_col] : 0.0f;
        }
        for(int i = tid; i < BK*BN; i += ts) {
            int row = i / BN;
            int col = i - row*BN;
            int b_row = k + row;
            int b_col = block_col + col;
            Bs[row][col] = (b_row < K && b_col < N) ? B[b_row*N + b_col] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for(int x = 0; x < BK; x++) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for(int i = 0; i < TM; i++)
                a_reg[i] = As[ty*TM+i][x];
            #pragma unroll
            for(int i = 0; i < TN; i++)
                b_reg[i] = Bs[x][tx*TN+i];
            #pragma unroll
            for(int i = 0; i < TM; i++)
                for(int j = 0; j < TN; j++)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i = 0; i < TM; i++) {
        int row = thread_row0 + i;
        if(row >= M) continue;
        #pragma unroll
        for(int j = 0; j < TN; j++) {
            int col = thread_col0 + j;
            if(col >= N) continue;
            C[row*N + col] = acc[i][j];
        }
    }
}

// Requires: BM = blockDim.y * TM, BN = blockDim.x * TN
template<int BM, int BK, int BN, int TM, int TN>
__global__ void GEMMThreadTileFloat4(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M,
                               int N, 
                               int K) {
    static_assert(BM % TM == 0);
    static_assert(BN % TN == 0);
    static_assert(BN % 4 == 0);
    static_assert(BK % 4 == 0);
    static_assert(TN % 4 == 0);
    assert(N % 4 == 0);
    assert(K % 4 == 0);
    assert(M % BM == 0);
    assert(N % BN == 0);
    assert(K % BK == 0);
    assert(blockDim.x * TN == BN);
    assert(blockDim.y * TM == BM);
    __shared__ float As[BM][BK], Bs[BK][BN];
    int tx = threadIdx.x, ty = threadIdx.y;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int thread_row0 = block_row + ty * TM;
    int thread_col0 = block_col + tx * TN;
    float acc[TM][TN] = {0.0f};
    for(int k = 0; k < K; k += BK) {
        int tid = tx + ty * blockDim.x;
        int ts = blockDim.x * blockDim.y;
        for(int i = tid; i < BM*BK/4; i += ts) {
            int row = i*4 / BK;
            int col = i*4 - row*BK;
            int a_row = block_row + row;
            int a_col = k + col;
            float4 v = *reinterpret_cast<const float4*>(&A[a_row*K + a_col]);
            *reinterpret_cast<float4*>(&As[row][col]) = v;
        }
        for(int i = tid; i < BK*BN/4; i += ts) {
            int row = i*4 / BN;
            int col = i*4 - row*BN;
            int b_row = k + row;
            int b_col = block_col + col;
            float4 v = *reinterpret_cast<const float4*>(&B[b_row*N + b_col]);
            *reinterpret_cast<float4*>(&Bs[row][col]) = v;
        }
        __syncthreads();
        #pragma unroll
        for(int x = 0; x < BK; x++) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for(int i = 0; i < TM; i++)
                a_reg[i] = As[ty*TM+i][x];
            #pragma unroll
            for(int i = 0; i < TN; i++)
                b_reg[i] = Bs[x][tx*TN+i];
            #pragma unroll
            for(int i = 0; i < TM; i++)
                for(int j = 0; j < TN; j++)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i = 0; i < TM; i++) {
        int row = thread_row0 + i;
        if(row >= M) continue;
        #pragma unroll
        for(int j = 0; j < TN/4; j++) {
            int col = thread_col0 + j*4;
            if(col >= N) continue;
            int jj = j * 4;
            float4 v = make_float4(acc[i][jj], acc[i][jj+1], acc[i][jj+2], acc[i][jj+3]);
            *reinterpret_cast<float4*>(&C[row*N + col]) = v;
        }
    }
}