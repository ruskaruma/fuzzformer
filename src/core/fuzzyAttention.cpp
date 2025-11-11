#include "fuzzformer/fuzzyAttention.h"

#ifdef FUZZFORMER_HAS_TORCH

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "fuzzformer/tensorUtils.h"

namespace fuzzformer {

namespace kernels {
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
                                    cudaStream_t stream);
}  // namespace kernels

namespace {

void check_tensor(const torch::Tensor& tensor,
                  const char* name,
                  torch::ScalarType expected_type) {
  tensor::ensure_cuda(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == expected_type,
              name, " must be of type ", expected_type);
  TORCH_CHECK(tensor.dim() == 4,
              name, " must have shape [batch, heads, seq_len, head_dim]");
}

void check_parameter(const torch::Tensor& tensor,
                     const char* name,
                     int64_t expected_size,
                     torch::ScalarType expected_type) {
  tensor::ensure_cuda(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == expected_type,
              name, " must be of type ", expected_type);
  TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == expected_size,
              name, " must have shape [num_heads]");
}

}  // namespace

torch::Tensor fuzzy_attention_forward(const torch::Tensor& queries,
                                      const torch::Tensor& keys,
                                      const torch::Tensor& values,
                                      const torch::Tensor& alpha,
                                      const torch::Tensor& beta) {
  auto q = queries.contiguous();
  auto k = keys.contiguous();
  auto v = values.contiguous();

  check_tensor(q, "queries", torch::kFloat32);
  check_tensor(k, "keys", torch::kFloat32);
  check_tensor(v, "values", torch::kFloat32);

  const auto batch_size = q.size(0);
  const auto num_heads = q.size(1);
  const auto seq_len = q.size(2);
  const auto head_dim = q.size(3);

  TORCH_CHECK(k.sizes() == q.sizes(), "keys must match queries shape");
  TORCH_CHECK(v.sizes() == q.sizes(), "values must match queries shape");

  tensor::validate_attention_dims(batch_size, num_heads, seq_len, head_dim);

  auto alpha_vec = alpha.contiguous();
  auto beta_vec = beta.contiguous();
  check_parameter(alpha_vec, "alpha", num_heads, torch::kFloat32);
  check_parameter(beta_vec, "beta", num_heads, torch::kFloat32);

  auto output = torch::empty_like(q);

  kernels::launch_fuzzy_attention_forward(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      v.data_ptr<float>(),
      alpha_vec.data_ptr<float>(),
      beta_vec.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<int>(batch_size),
      static_cast<int>(num_heads),
      static_cast<int>(seq_len),
      static_cast<int>(head_dim),
      at::cuda::getCurrentCUDAStream());

  const auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "fuzzy_attention_forward kernel launch failed: ",
              cudaGetErrorString(err));

  return output;
}

std::vector<torch::Tensor> fuzzy_attention_backward(
    const torch::Tensor& grad_out,
    const FuzzyAttentionContext& /*context*/) {
  TORCH_CHECK(false, "fuzzy_attention_backward is not implemented yet");
  return {};
}

}  // namespace fuzzformer

#else  // FUZZFORMER_HAS_TORCH

namespace fuzzformer {

torch::Tensor fuzzy_attention_forward(const torch::Tensor&,
                                      const torch::Tensor&,
                                      const torch::Tensor&,
                                      const torch::Tensor&,
                                      const torch::Tensor&) {
  return {};
}

std::vector<torch::Tensor> fuzzy_attention_backward(
    const torch::Tensor&,
    const FuzzyAttentionContext&) {
  return {};
}

}  // namespace fuzzformer

#endif  // FUZZFORMER_HAS_TORCH

