#include <cuda_runtime.h>
#include <math_constants.h>

namespace fuzzformer {
namespace kernels {

namespace {

constexpr float kEpsilon = 1e-6f;

__global__ void fuzzy_attention_forward_kernel(const float* __restrict__ queries,
                                               const float* __restrict__ keys,
                                               const float* __restrict__ values,
                                               const float* __restrict__ alpha,
                                               const float* __restrict__ beta,
                                               float* __restrict__ output,
                                               int batch_size,
                                               int num_heads,
                                               int seq_len,
                                               int head_dim) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_rows = batch_size * num_heads * seq_len;
  if (row >= total_rows) {
    return;
  }

  const int bh = row / seq_len;
  const int query_index = row % seq_len;
  const int batch_index = bh / num_heads;
  const int head_index = bh % num_heads;

  const int base_offset = ((batch_index * num_heads + head_index) * seq_len + query_index) * head_dim;
  const float* q_vec = queries + base_offset;
  float* out_vec = output + base_offset;

  const float* k_head = keys + ((batch_index * num_heads + head_index) * seq_len * head_dim);
  const float* v_head = values + ((batch_index * num_heads + head_index) * seq_len * head_dim);

  const float alpha_h = alpha[head_index];
  const float beta_h = beta[head_index];

  const float scale = 1.0f / static_cast<float>(head_dim);

  float norm = 0.0f;
  for (int key_index = 0; key_index < seq_len; ++key_index) {
    const float* k_vec = k_head + key_index * head_dim;
    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      score += q_vec[d] * k_vec[d];
    }
    score *= scale;
    const float diff = score - beta_h;
    const float membership = __expf(-alpha_h * diff * diff);
    norm += membership;
  }

  float inv_norm = 0.0f;
  if (norm > kEpsilon) {
    inv_norm = 1.0f / norm;
  }

  for (int d = 0; d < head_dim; ++d) {
    out_vec[d] = 0.0f;
  }

  for (int key_index = 0; key_index < seq_len; ++key_index) {
    const float* k_vec = k_head + key_index * head_dim;
    const float* v_vec = v_head + key_index * head_dim;

    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      score += q_vec[d] * k_vec[d];
    }
    score *= scale;
    const float diff = score - beta_h;
    const float membership = __expf(-alpha_h * diff * diff);
    const float weight = membership * inv_norm;

    for (int d = 0; d < head_dim; ++d) {
      out_vec[d] += weight * v_vec[d];
    }
  }
}

}  // namespace

void launch_fuzzy_attention_forward(const float* queries,
                                    const float* keys,
                                    const float* values,
                                    const float* alpha,
                                    const float* beta,
                                    float* output,
                                    int batch_size,
                                    int num_heads,
                                    int seq_len,
                                    int head_dim,
                                    cudaStream_t stream) {
  const int total_rows = batch_size * num_heads * seq_len;
  const int threads = 128;
  const int blocks = (total_rows + threads - 1) / threads;
  fuzzy_attention_forward_kernel<<<blocks, threads, 0, stream>>>(
      queries,
      keys,
      values,
      alpha,
      beta,
      output,
      batch_size,
      num_heads,
      seq_len,
      head_dim);
}

}  // namespace kernels
}  // namespace fuzzformer

