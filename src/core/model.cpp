#include "fuzzformer/model.h"

#ifdef FUZZFORMER_HAS_TORCH

#include <utility>

namespace fuzzformer {

namespace {

torch::nn::Linear make_linear(std::size_t in, std::size_t out) {
  auto options = torch::nn::LinearOptions(in, out).bias(true);
  return torch::nn::Linear(options);
}

}  // namespace

FuzzFormerImpl::FuzzFormerImpl(ModelConfig config)
    : config_(std::move(config)),
      output_head_(register_module("output_head", make_linear(config_.model_dim, config_.model_dim))) {
  for (std::size_t i = 0; i < config_.num_layers; ++i) {
    auto block = TransformerBlock(config_);
    blocks_.push_back(register_module("block_" + std::to_string(i), block));
  }
}

torch::Tensor FuzzFormerImpl::forward(const torch::Tensor& input) {
  auto hidden = input;
  for (auto& block : blocks_) {
    hidden = block->forward(hidden);
  }
  return output_head_(hidden);
}

}  // namespace fuzzformer

#else

namespace fuzzformer {

FuzzFormerImpl::FuzzFormerImpl(ModelConfig config) : config_(std::move(config)), output_head_() {}

torch::Tensor FuzzFormerImpl::forward(const torch::Tensor& input) {
  return input;
}

}  // namespace fuzzformer

#endif



