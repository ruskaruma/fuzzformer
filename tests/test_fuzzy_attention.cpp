#include <gtest/gtest.h>

#include "fuzzformer/fuzzyAttention.h"
#include "fuzzformer/modelConfig.h"

#ifdef FUZZFORMER_HAS_TORCH
#include <torch/torch.h>
#endif

namespace fuzzformer {

TEST(FuzzyAttentionTest, ForwardProducesCorrectShape) {
#ifdef FUZZFORMER_HAS_TORCH
  const int batch_size = 2;
  const int num_heads = 4;
  const int seq_len = 8;
  const int head_dim = 64;

  auto q = torch::randn({batch_size, num_heads, seq_len, head_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto k = torch::randn({batch_size, num_heads, seq_len, head_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto v = torch::randn({batch_size, num_heads, seq_len, head_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto alpha = torch::ones({num_heads}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto beta = torch::zeros({num_heads}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  auto output = fuzzy_attention_forward(q, k, v, alpha, beta);
  ASSERT_EQ(output.sizes(), q.sizes());
#else
  GTEST_SKIP() << "libtorch not available";
#endif
}

TEST(FuzzyAttentionTest, BackwardProducesGradients) {
#ifdef FUZZFORMER_HAS_TORCH
  const int batch_size = 1;
  const int num_heads = 2;
  const int seq_len = 4;
  const int head_dim = 32;

  auto q = torch::randn({batch_size, num_heads, seq_len, head_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));
  auto k = torch::randn({batch_size, num_heads, seq_len, head_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));
  auto v = torch::randn({batch_size, num_heads, seq_len, head_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));
  auto alpha = torch::ones({num_heads}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));
  auto beta = torch::zeros({num_heads}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));

  FuzzyAttentionContext context{q, k, v, alpha, beta};
  auto output = fuzzy_attention_forward(q, k, v, alpha, beta);
  auto grad_out = torch::ones_like(output);

  auto grads = fuzzy_attention_backward(grad_out, context);
  ASSERT_EQ(grads.size(), 5);
  ASSERT_EQ(grads[0].sizes(), q.sizes());
  ASSERT_EQ(grads[1].sizes(), k.sizes());
  ASSERT_EQ(grads[2].sizes(), v.sizes());
#else
  GTEST_SKIP() << "libtorch not available";
#endif
}

}  // namespace fuzzformer

