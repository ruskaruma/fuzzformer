#pragma once

#ifdef FUZZFORMER_HAS_TORCH
#include <torch/torch.h>
#else
#include "fuzzformer/torchStub.h"
#endif

#include "fuzzformer/fuzzyAttention.h"
#include "fuzzformer/modelConfig.h"

namespace fuzzformer {

class TransformerBlockImpl : public torch::nn::Module {
 public:
  explicit TransformerBlockImpl(ModelConfig config);

  torch::Tensor forward(const torch::Tensor& input);

 private:
  ModelConfig config_;
  torch::nn::Linear q_proj_;
  torch::nn::Linear k_proj_;
  torch::nn::Linear v_proj_;
  torch::nn::Linear out_proj_;
};

TORCH_MODULE(TransformerBlock);

}  // namespace fuzzformer

