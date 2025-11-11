#include "fuzzformer/fuzzyAttention.h"

namespace fuzzformer {

#ifdef FUZZFORMER_HAS_TORCH

torch::Tensor fuzzy_attention_forward(const torch::Tensor& queries,
                                      const torch::Tensor& keys,
                                      const torch::Tensor& values,
                                      const torch::Tensor& alpha,
                                      const torch::Tensor& beta) {
  TORCH_CHECK(queries.device().is_cuda(), "fuzzy_attention_forward expects CUDA tensors");
  TORCH_CHECK(false, "fuzzy_attention_forward is not implemented yet");
  return torch::Tensor();
}

std::vector<torch::Tensor> fuzzy_attention_backward(
    const torch::Tensor& grad_out,
    const FuzzyAttentionContext& context) {
  TORCH_CHECK(grad_out.device().is_cuda(), "fuzzy_attention_backward expects CUDA tensors");
  TORCH_CHECK(false, "fuzzy_attention_backward is not implemented yet");
  return {};
}

#else

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

#endif

}  // namespace fuzzformer

