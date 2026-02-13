/*
CUDA Attention Interview Series (scaled dot-product attention)

File: attention_0_naive_fused.cu
Focus:
  - Compute attention for a single batch (B=1), multiple heads.
  - One CUDA block computes ONE (head, query_position) output vector.
  - "Fused but naive":
    - compute all scores q·k
    - reduce max + sum(exp(.)) for softmax
    - compute output = softmax(scores) @ V
  - Uses shared memory to store per-key exp(score - max).

Interview tags:
  - Classic:     Very High
  - Importance:  Extremely High (LLM inference core)
  - Frequency:   Very High

Memorize:
  - Not the final version. This is the cleanest correctness-first baseline.

Build (recommended for RTX 4070 / SM89):
  nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 attention_0_naive_fused.cu \
    -o attention_0_naive_fused
Run:
  ./attention_0_naive_fused [seq_len<=256] [head_dim] [num_heads] \
    [causal(0|1)] [device_id]
Example:
  ./attention_0_naive_fused 128 64 8 1 0

Typical interviewer follow-ups:
  - "Don't materialize the softmax vector": online softmax.
  - Blockwise / tiled K,V: FlashAttention-style.
  - GQA / MQA and kv-cache layout.
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kDefaultSeqLen = 128;
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

__global__ void AttentionKernelNaiveFused(const float* __restrict__ q,
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

  const int query_i = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  if (head >= num_heads || query_i >= seq_len) {
    return;
  }

  const int key_j = threadIdx.x;
  float score = -INFINITY;
  if (key_j < seq_len && (!causal || (key_j <= query_i))) {
    float dot = 0.0f;
    const size_t q_base =
        (static_cast<size_t>(head) * static_cast<size_t>(seq_len) +
         static_cast<size_t>(query_i)) *
        static_cast<size_t>(head_dim);
    const size_t k_base =
        (static_cast<size_t>(head) * static_cast<size_t>(seq_len) +
         static_cast<size_t>(key_j)) *
        static_cast<size_t>(head_dim);
    for (int d = 0; d < head_dim; ++d) {
      dot += q[q_base + static_cast<size_t>(d)] *
             k[k_base + static_cast<size_t>(d)];
    }
    score = dot * scale;
  }

  const float row_max = BlockReduceMax(smem, score);
  const float p = (score == -INFINITY) ? 0.0f : __expf(score - row_max);
  p_smem[key_j] = p;
  __syncthreads();

  const float row_sum = BlockReduceSum(smem, p);
  const float inv = (row_sum == 0.0f) ? 0.0f : (1.0f / row_sum);
  __syncthreads();

  // Compute output vector. Use the first head_dim threads.
  const int d = threadIdx.x;
  if (d < head_dim) {
    float out = 0.0f;
    const size_t v_head_base =
        static_cast<size_t>(head) * static_cast<size_t>(seq_len) *
        static_cast<size_t>(head_dim);
    for (int j = 0; j < seq_len; ++j) {
      const float pj = p_smem[j] * inv;
      const size_t v_idx = v_head_base +
                           static_cast<size_t>(j) *
                               static_cast<size_t>(head_dim) +
                           static_cast<size_t>(d);
      out += pj * v[v_idx];
    }
    const size_t o_idx =
        (static_cast<size_t>(head) * static_cast<size_t>(seq_len) +
         static_cast<size_t>(query_i)) *
            static_cast<size_t>(head_dim) +
        static_cast<size_t>(d);
    o[o_idx] = out;
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
  if (seq_len > kBlockSize) {
    std::fprintf(stderr,
                 "This baseline stores per-key probs in shared memory.\n"
                 "Please run with seq_len <= %d (got %d).\n",
                 kBlockSize, seq_len);
    return 2;
  }
  if (head_dim > kBlockSize) {
    std::fprintf(stderr,
                 "This baseline uses the first head_dim threads to write the\n"
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
  AttentionKernelNaiveFused<<<grid, block>>>(d_q, d_k, d_v, d_o, num_heads,
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
