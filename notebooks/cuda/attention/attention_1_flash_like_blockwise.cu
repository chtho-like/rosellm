/*
CUDA Attention Interview Series (FlashAttention-style blockwise softmax)

File: attention_1_flash_like_blockwise.cu
Focus:
  - Same attention math as attention_0.
  - Process keys in blocks and use the online softmax update:
      m_new = max(m, max(scores_block))
      l_new = l * exp(m - m_new) + sum(exp(scores_block - m_new))
      o_new = o * exp(m - m_new) + sum(exp(scores_block - m_new) * V_block)
    Then output = o / l.
  - This avoids materializing the softmax vector for the full seq_len.

Interview tags:
  - Classic:     Very High
  - Importance:  Extremely High
  - Frequency:   Very High (for LLM inference roles)

Memorize (recommended):
  - YES. This is a compact "FlashAttention core idea" you can hand-write.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 attention_1_flash_like_blockwise.cu \
    -o attention_1_flash_like_blockwise
Run:
  ./attention_1_flash_like_blockwise [seq_len] [head_dim] [num_heads] \
    [causal(0|1)] [device_id]
Example:
  ./attention_1_flash_like_blockwise 1024 64 8 1 0

Notes:
  - This is still far from production FlashAttention:
    - no K/V tiling into shared
    - no tensor cores
    - no query tiling (one query per block)
    But the numerically stable online softmax is the key interview concept.
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kDefaultSeqLen = 1024;
constexpr int kDefaultHeadDim = 64;
constexpr int kDefaultNumHeads = 8;

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

void FillQkv(std::vector<float>* q,
             std::vector<float>* k,
             std::vector<float>* v,
             int num_heads,
             int seq_len,
             int head_dim) {
  const size_t n = static_cast<size_t>(num_heads) *
                   static_cast<size_t>(seq_len) *
                   static_cast<size_t>(head_dim);
  q->resize(n);
  k->resize(n);
  v->resize(n);
  for (int h = 0; h < num_heads; ++h) {
    for (int i = 0; i < seq_len; ++i) {
      for (int d = 0; d < head_dim; ++d) {
        const int base = (h + 1) * 100000 + (i + 1) * 1000 + (d + 1);
        const float fq = static_cast<float>((base * 17) % 29 - 14) * 0.01f;
        const float fk = static_cast<float>((base * 13) % 31 - 15) * 0.01f;
        const float fv = static_cast<float>((base * 11) % 37 - 18) * 0.01f;
        const size_t idx =
            (static_cast<size_t>(h) * static_cast<size_t>(seq_len) +
             static_cast<size_t>(i)) *
                static_cast<size_t>(head_dim) +
            static_cast<size_t>(d);
        (*q)[idx] = fq;
        (*k)[idx] = fk;
        (*v)[idx] = fv;
      }
    }
  }
}

void CpuAttention(const std::vector<float>& q,
                  const std::vector<float>& k,
                  const std::vector<float>& v,
                  std::vector<float>* o,
                  int num_heads,
                  int seq_len,
                  int head_dim,
                  bool causal) {
  const size_t n = static_cast<size_t>(num_heads) *
                   static_cast<size_t>(seq_len) *
                   static_cast<size_t>(head_dim);
  o->assign(n, 0.0f);
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

  std::vector<double> scores(static_cast<size_t>(seq_len));
  std::vector<double> probs(static_cast<size_t>(seq_len));

  for (int h = 0; h < num_heads; ++h) {
    for (int i = 0; i < seq_len; ++i) {
      double max_s = -INFINITY;
      for (int j = 0; j < seq_len; ++j) {
        if (causal && (j > i)) {
          scores[static_cast<size_t>(j)] = -INFINITY;
          continue;
        }
        double dot = 0.0;
        const size_t q_base =
            (static_cast<size_t>(h) * static_cast<size_t>(seq_len) +
             static_cast<size_t>(i)) *
            static_cast<size_t>(head_dim);
        const size_t k_base =
            (static_cast<size_t>(h) * static_cast<size_t>(seq_len) +
             static_cast<size_t>(j)) *
            static_cast<size_t>(head_dim);
        for (int d = 0; d < head_dim; ++d) {
          dot += static_cast<double>(q[q_base + static_cast<size_t>(d)]) *
                 static_cast<double>(k[k_base + static_cast<size_t>(d)]);
        }
        const double s = dot * scale;
        scores[static_cast<size_t>(j)] = s;
        max_s = std::max(max_s, s);
      }

      double sum = 0.0;
      for (int j = 0; j < seq_len; ++j) {
        const double s = scores[static_cast<size_t>(j)];
        if (!std::isfinite(s)) {
          probs[static_cast<size_t>(j)] = 0.0;
          continue;
        }
        const double p = std::exp(s - max_s);
        probs[static_cast<size_t>(j)] = p;
        sum += p;
      }
      const double inv = (sum == 0.0) ? 0.0 : (1.0 / sum);

      for (int d = 0; d < head_dim; ++d) {
        double out = 0.0;
        const size_t v_head_base =
            static_cast<size_t>(h) * static_cast<size_t>(seq_len) *
            static_cast<size_t>(head_dim);
        for (int j = 0; j < seq_len; ++j) {
          const double pj = probs[static_cast<size_t>(j)] * inv;
          out += pj * static_cast<double>(
                          v[v_head_base + static_cast<size_t>(j) *
                                               static_cast<size_t>(head_dim) +
                            static_cast<size_t>(d)]);
        }
        const size_t o_idx =
            (static_cast<size_t>(h) * static_cast<size_t>(seq_len) +
             static_cast<size_t>(i)) *
                static_cast<size_t>(head_dim) +
            static_cast<size_t>(d);
        (*o)[o_idx] = static_cast<float>(out);
      }
    }
  }
}

__device__ __forceinline__ float BlockReduceMax(float* smem, float v) {
  const int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }
  return smem[0];
}

__device__ __forceinline__ float BlockReduceSum(float* smem, float v) {
  const int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  return smem[0];
}

__global__ void AttentionKernelFlashLike(const float* __restrict__ q,
                                         const float* __restrict__ k,
                                         const float* __restrict__ v,
                                         float* __restrict__ o,
                                         int num_heads,
                                         int seq_len,
                                         int head_dim,
                                         float scale,
                                         bool causal) {
  __shared__ float smem[kBlockSize];
  __shared__ float p_smem[kBlockSize];
  __shared__ float shared_alpha;
  __shared__ float shared_l;
  __shared__ float shared_inv_l;

  const int query_i = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  if (head >= num_heads || query_i >= seq_len) {
    return;
  }

  // Thread 0 maintains the online softmax scalars.
  float m = -INFINITY;
  float l = 0.0f;

  // One output scalar per dimension, stored in the first head_dim threads.
  float out = 0.0f;

  const size_t q_base =
      (static_cast<size_t>(head) * static_cast<size_t>(seq_len) +
       static_cast<size_t>(query_i)) *
      static_cast<size_t>(head_dim);
  const size_t kv_head_base =
      static_cast<size_t>(head) * static_cast<size_t>(seq_len) *
      static_cast<size_t>(head_dim);

  for (int kb = 0; kb < seq_len; kb += kBlockSize) {
    const int key_j = kb + threadIdx.x;
    const bool valid = (key_j < seq_len) && (!causal || (key_j <= query_i));

    float score = -INFINITY;
    if (valid) {
      float dot = 0.0f;
      const size_t k_base = kv_head_base +
                            static_cast<size_t>(key_j) *
                                static_cast<size_t>(head_dim);
      for (int d = 0; d < head_dim; ++d) {
        dot += q[q_base + static_cast<size_t>(d)] *
               k[k_base + static_cast<size_t>(d)];
      }
      score = dot * scale;
    }

    const float block_max = BlockReduceMax(smem, score);
    if (threadIdx.x == 0) {
      if (block_max == -INFINITY) {
        shared_alpha = 1.0f;
      } else if (m == -INFINITY) {
        m = block_max;
        shared_alpha = 0.0f;
      } else {
        const float m_new = fmaxf(m, block_max);
        shared_alpha = __expf(m - m_new);
        m = m_new;
      }
      // Broadcast m through shared memory: reuse smem[0].
      smem[0] = m;
    }
    __syncthreads();

    const float m_new = smem[0];
    const float alpha = shared_alpha;
    const float p =
        (valid && (m_new != -INFINITY)) ? __expf(score - m_new) : 0.0f;
    p_smem[threadIdx.x] = p;
    __syncthreads();

    const float sum_p = BlockReduceSum(smem, p);
    if (threadIdx.x == 0) {
      l = l * alpha + sum_p;
      shared_l = l;
    }
    __syncthreads();

    const int d = threadIdx.x;
    if (d < head_dim) {
      float delta = 0.0f;
      for (int t = 0; t < kBlockSize; ++t) {
        const int key = kb + t;
        if (key >= seq_len) {
          break;
        }
        const float pj = p_smem[t];
        const size_t v_idx = kv_head_base +
                             static_cast<size_t>(key) *
                                 static_cast<size_t>(head_dim) +
                             static_cast<size_t>(d);
        delta += pj * v[v_idx];
      }
      out = out * alpha + delta;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    shared_inv_l = (l == 0.0f) ? 0.0f : (1.0f / l);
  }
  __syncthreads();

  const int d = threadIdx.x;
  if (d < head_dim) {
    const float out_final = out * shared_inv_l;
    const size_t o_idx =
        (static_cast<size_t>(head) * static_cast<size_t>(seq_len) +
         static_cast<size_t>(query_i)) *
            static_cast<size_t>(head_dim) +
        static_cast<size_t>(d);
    o[o_idx] = out_final;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int seq_len =
      (argc >= 2) ? ParseInt(argv[1], kDefaultSeqLen) : kDefaultSeqLen;
  const int head_dim =
      (argc >= 3) ? ParseInt(argv[2], kDefaultHeadDim) : kDefaultHeadDim;
  const int num_heads =
      (argc >= 4) ? ParseInt(argv[3], kDefaultNumHeads) : kDefaultNumHeads;
  const bool causal = (argc >= 5) ? (ParseInt(argv[4], 0) != 0) : true;
  const int device_id = (argc >= 6) ? ParseInt(argv[5], 0) : 0;

  if (seq_len <= 0 || head_dim <= 0 || num_heads <= 0) {
    std::fprintf(stderr, "seq_len, head_dim, num_heads must be > 0\n");
    return 2;
  }
  if (head_dim > kBlockSize) {
    std::fprintf(stderr,
                 "This demo uses the first head_dim threads to write the\n"
                 "output. Please run with head_dim <= %d (got %d).\n",
                 kBlockSize, head_dim);
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(device_id));

  std::vector<float> h_q;
  std::vector<float> h_k;
  std::vector<float> h_v;
  FillQkv(&h_q, &h_k, &h_v, num_heads, seq_len, head_dim);

  std::vector<float> h_o_cpu;
  CpuAttention(h_q, h_k, h_v, &h_o_cpu, num_heads, seq_len, head_dim, causal);

  const size_t n = static_cast<size_t>(num_heads) *
                   static_cast<size_t>(seq_len) *
                   static_cast<size_t>(head_dim);
  const size_t bytes = n * sizeof(float);

  float* d_q = nullptr;
  float* d_k = nullptr;
  float* d_v = nullptr;
  float* d_o = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, bytes));
  CUDA_CHECK(cudaMalloc(&d_k, bytes));
  CUDA_CHECK(cudaMalloc(&d_v, bytes));
  CUDA_CHECK(cudaMalloc(&d_o, bytes));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice));

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  const dim3 block(kBlockSize, 1, 1);
  const dim3 grid(static_cast<unsigned int>(seq_len),
                  static_cast<unsigned int>(num_heads), 1);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  AttentionKernelFlashLike<<<grid, block>>>(d_q, d_k, d_v, d_o, num_heads,
                                            seq_len, head_dim, scale, causal);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<float> h_o_gpu(n);
  CUDA_CHECK(cudaMemcpy(h_o_gpu.data(), d_o, bytes, cudaMemcpyDeviceToHost));

  double max_abs_err = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double diff = std::abs(static_cast<double>(h_o_gpu[i]) -
                                 static_cast<double>(h_o_cpu[i]));
    max_abs_err = std::max(max_abs_err, diff);
  }

  std::printf("seq_len:      %d\n", seq_len);
  std::printf("head_dim:     %d\n", head_dim);
  std::printf("num_heads:    %d\n", num_heads);
  std::printf("causal:       %d\n", causal ? 1 : 0);
  std::printf("time_ms:      %.3f\n", ms);
  std::printf("max_abs_err:  %.6g\n", max_abs_err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));

  return (max_abs_err <= 1e-4) ? 0 : 1;
}
