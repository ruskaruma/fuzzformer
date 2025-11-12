#include "fuzzformer/transformerBlock.h"

#include <cmath>

namespace fuzzformer {

#ifdef FUZZFORMER_HAS_TORCH

namespace {

torch::nn::Linear make_linear(std::size_t in, std::size_t out) {
  auto options = torch::nn::LinearOptions(in, out).bias(true);
  return torch::nn::Linear(options);
}

torch::Tensor reshape_heads(const torch::Tensor& tensor,
                            int64_t batch,
                            int64_t seq_len,
                            int64_t num_heads,
                            int64_t head_dim) {
  return tensor.view({batch, seq_len, num_heads, head_dim})
      .permute({0, 2, 1, 3})
      .contiguous();
}

}  // namespace

TransformerBlockImpl::TransformerBlockImpl(ModelConfig config)
    : config_(std::move(config)),
      q_proj_(register_module("q_proj", make_linear(config_.model_dim, config_.model_dim))),
      k_proj_(register_module("k_proj", make_linear(config_.model_dim, config_.model_dim))),
      v_proj_(register_module("v_proj", make_linear(config_.model_dim, config_.model_dim))),
      out_proj_(register_module("out_proj", make_linear(config_.model_dim, config_.model_dim))) {}

torch::Tensor TransformerBlockImpl::forward(const torch::Tensor& input) {
  TORCH_CHECK(input.dim() == 3, "TransformerBlock expects input of shape [batch, seq_len, model_dim]");

  const auto batch = input.size(0);
  const auto seq_len = input.size(1);
  const auto model_dim = input.size(2);

  TORCH_CHECK(model_dim == static_cast<int64_t>(config_.model_dim),
              "Input dim mismatch: expected ",
              config_.model_dim,
              " got ",
              model_dim);
  TORCH_CHECK(config_.num_heads > 0, "num_heads must be positive");
  TORCH_CHECK(model_dim % static_cast<int64_t>(config_.num_heads) == 0,
              "model_dim must be divisible by num_heads");

  const auto head_dim = model_dim / static_cast<int64_t>(config_.num_heads);

  auto q_proj = q_proj_(input);
  auto k_proj = k_proj_(input);
  auto v_proj = v_proj_(input);

  auto q_heads = reshape_heads(q_proj, batch, seq_len, config_.num_heads, head_dim);
  auto k_heads = reshape_heads(k_proj, batch, seq_len, config_.num_heads, head_dim);
  auto v_heads = reshape_heads(v_proj, batch, seq_len, config_.num_heads, head_dim);

  auto alpha = torch::ones({static_cast<int64_t>(config_.num_heads)}, q_heads.options());
  auto beta = torch::zeros({static_cast<int64_t>(config_.num_heads)}, q_heads.options());

  auto attn = fuzzy_attention_forward(q_heads, k_heads, v_heads, alpha, beta);
  auto merged = attn.permute({0, 2, 1, 3}).contiguous().view({batch, seq_len, model_dim});

  auto output = out_proj_(merged);
  return output + input;
}

#else  // FUZZFORMER_HAS_TORCH

TransformerBlockImpl::TransformerBlockImpl(ModelConfig config) : config_(std::move(config)) {}

torch::Tensor TransformerBlockImpl::forward(const torch::Tensor& input) {
  return input;
}

#endif  // FUZZFORMER_HAS_TORCH

}  // namespace fuzzformer

