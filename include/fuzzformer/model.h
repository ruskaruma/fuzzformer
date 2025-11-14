#pragma once

#ifdef FUZZFORMER_HAS_TORCH
#include <torch/torch.h>
#else
#include "fuzzformer/torchStub.h"
#endif

#include "fuzzformer/modelConfig.h"
#include "fuzzformer/transformerBlock.h"

namespace fuzzformer {

class FuzzFormerImpl : public torch::nn::Module {
 public:
  explicit FuzzFormerImpl(ModelConfig config);

  torch::Tensor forward(const torch::Tensor& input);

 private:
  ModelConfig config_;
  std::vector<TransformerBlock> blocks_;
  torch::nn::Linear output_head_;
};

TORCH_MODULE(FuzzFormer);

}  // namespace fuzzformer



