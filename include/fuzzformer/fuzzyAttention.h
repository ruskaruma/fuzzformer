#pragma once

#ifdef FUZZFORMER_HAS_TORCH
#include <torch/torch.h>
#else
#include "fuzzformer/torchStub.h"
#endif

namespace fuzzformer {

torch::Tensor fuzzy_attention_forward(const torch::Tensor& queries,
                                      const torch::Tensor& keys,
                                      const torch::Tensor& values,
                                      const torch::Tensor& alpha,
                                      const torch::Tensor& beta);

struct FuzzyAttentionContext {
  torch::Tensor queries;
  torch::Tensor keys;
  torch::Tensor values;
  torch::Tensor alpha;
  torch::Tensor beta;
};

std::vector<torch::Tensor> fuzzy_attention_backward(
    const torch::Tensor& grad_out,
    const FuzzyAttentionContext& context);

}  // namespace fuzzformer

