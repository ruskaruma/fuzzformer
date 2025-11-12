#include <cuda_runtime.h>
#include <math_constants.h>

namespace fuzzformer {
namespace kernels {

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

__global__ void fuzzy_attention_backward_kernel(const float* __restrict__ grad_out,
                                                const float* __restrict__ queries,
                                                const float* __restrict__ keys,
                                                const float* __restrict__ values,
                                                const float* __restrict__ alpha,
                                                const float* __restrict__ beta,
                                                float* __restrict__ d_queries,
                                                float* __restrict__ d_keys,
                                                float* __restrict__ d_values,
                                                float* __restrict__ d_alpha,
                                                float* __restrict__ d_beta,
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

  const int row_offset = ((batch_index * num_heads + head_index) * seq_len + query_index) * head_dim;
  const float* q_vec = queries + row_offset;
  const float* grad_vec = grad_out + row_offset;
  float* dq_vec = d_queries + row_offset;

  const int head_base = (batch_index * num_heads + head_index) * seq_len * head_dim;
  const float* k_head = keys + head_base;
  const float* v_head = values + head_base;
  float* dk_head = d_keys + head_base;
  float* dv_head = d_values + head_base;

  const float alpha_h = alpha[head_index];
  const float beta_h = beta[head_index];

  for (int d = 0; d < head_dim; ++d) {
    dq_vec[d] = 0.0f;
  }

  float norm = 0.0f;
  for (int key_index = 0; key_index < seq_len; ++key_index) {
    const float* k_vec = k_head + key_index * head_dim;
    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      score += q_vec[d] * k_vec[d];
    }
    score /= static_cast<float>(head_dim);
    const float diff = score - beta_h;
    const float membership = __expf(-alpha_h * diff * diff);
    norm += membership;
  }

  const float inv_norm = norm > kEpsilon ? 1.0f / norm : 0.0f;
  float sum_gw_w = 0.0f;

  for (int key_index = 0; key_index < seq_len; ++key_index) {
    const float* k_vec = k_head + key_index * head_dim;
    const float* v_vec = v_head + key_index * head_dim;
    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      score += q_vec[d] * k_vec[d];
    }
    score /= static_cast<float>(head_dim);
    const float diff = score - beta_h;
    const float membership = __expf(-alpha_h * diff * diff);
    const float weight = membership * inv_norm;

    float gw = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      gw += grad_vec[d] * v_vec[d];
      atomicAdd(dv_head + key_index * head_dim + d, weight * grad_vec[d]);
    }
    sum_gw_w += gw * weight;
  }

  float grad_alpha_accum = 0.0f;
  float grad_beta_accum = 0.0f;

  for (int key_index = 0; key_index < seq_len; ++key_index) {
    const float* k_vec = k_head + key_index * head_dim;
    const float* v_vec = v_head + key_index * head_dim;
    float* dk_vec = dk_head + key_index * head_dim;

    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      score += q_vec[d] * k_vec[d];
    }
    score /= static_cast<float>(head_dim);
    const float diff = score - beta_h;
    const float membership = __expf(-alpha_h * diff * diff);
    const float weight = membership * inv_norm;

    float gw = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      gw += grad_vec[d] * v_vec[d];
    }

    const float g_m = inv_norm > 0.0f ? (gw - sum_gw_w) * inv_norm : 0.0f;
    const float g_s = -2.0f * alpha_h * diff * membership * g_m;

    grad_alpha_accum += g_m * (-diff * diff * membership);
    grad_beta_accum += g_m * (2.0f * alpha_h * diff * membership);

    const float scale = 1.0f / static_cast<float>(head_dim);
    for (int d = 0; d < head_dim; ++d) {
      const float k_val = k_vec[d];
      const float q_val = q_vec[d];
      dq_vec[d] += g_s * k_val * scale;
      atomicAdd(dk_vec + d, g_s * q_val * scale);
    }
  }

  atomicAdd(d_alpha + head_index, grad_alpha_accum);
  atomicAdd(d_beta + head_index, grad_beta_accum);
}

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

void launch_fuzzy_attention_backward(const float* grad_out,
                                     const float* queries,
                                     const float* keys,
                                     const float* values,
                                     const float* alpha,
                                     const float* beta,
                                     float* d_queries,
                                     float* d_keys,
                                     float* d_values,
                                     float* d_alpha,
                                     float* d_beta,
                                     int batch_size,
                                     int num_heads,
                                     int seq_len,
                                     int head_dim,
                                     cudaStream_t stream) {
  const int total_rows = batch_size * num_heads * seq_len;
  const int threads = 128;
  const int blocks = (total_rows + threads - 1) / threads;
  fuzzy_attention_backward_kernel<<<blocks, threads, 0, stream>>>(
      grad_out,
      queries,
      keys,
      values,
      alpha,
      beta,
      d_queries,
      d_keys,
      d_values,
      d_alpha,
      d_beta,
      batch_size,
      num_heads,
      seq_len,
      head_dim);
}

}  // namespace kernels
}  // namespace fuzzformer
