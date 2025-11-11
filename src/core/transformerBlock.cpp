#include "fuzzformer/transformerBlock.h"

#include <cmath>

namespace fuzzformer {

#ifdef FUZZFORMER_HAS_TORCH

namespace {
torch::nn::Linear make_linear(std::size_t in, std::size_t out) {
  auto options = torch::nn::LinearOptions(in, out).bias(true);
  return torch::nn::Linear(options);
}
}  // namespace

TransformerBlockImpl::TransformerBlockImpl(ModelConfig config)
    : config_(std::move(config)),
      q_proj_(register_module("q_proj", make_linear(config_.model_dim, config_.model_dim))),
      k_proj_(register_module("k_proj", make_linear(config_.model_dim, config_.model_dim))),
      v_proj_(register_module("v_proj", make_linear(config_.model_dim, config_.model_dim))),
      out_proj_(register_module("out_proj", make_linear(config_.model_dim, config_.model_dim))) {}

torch::Tensor TransformerBlockImpl::forward(const torch::Tensor& input) {
  auto queries = q_proj_(input);
  auto keys = k_proj_(input);
  auto values = v_proj_(input);
  auto alpha = torch::ones({config_.num_heads}, queries.options());
  auto beta = torch::zeros({config_.num_heads}, queries.options());
  auto output = fuzzy_attention_forward(queries, keys, values, alpha, beta);
  return out_proj_(output);
}

#else

TransformerBlockImpl::TransformerBlockImpl(ModelConfig config) : config_(std::move(config)) {}

torch::Tensor TransformerBlockImpl::forward(const torch::Tensor& input) {
  return input;
}

#endif

}  // namespace fuzzformer

