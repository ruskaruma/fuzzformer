#include <gtest/gtest.h>

#include "fuzzformer/modelConfig.h"
#include "fuzzformer/transformerBlock.h"

#ifdef FUZZFORMER_HAS_TORCH
#include <torch/torch.h>
#endif

namespace fuzzformer {

TEST(TransformerBlockTest, PreservesInputShape) {
#ifdef FUZZFORMER_HAS_TORCH
  ModelConfig config;
  config.model_dim = 128;
  config.num_heads = 4;

  auto block = TransformerBlock(config);
  auto input = torch::ones({2, 8, static_cast<long>(config.model_dim)},
                           torch::TensorOptions().dtype(torch::kFloat32));
  auto output = block->forward(input);
  ASSERT_EQ(output.sizes(), input.sizes());
#else
  GTEST_SKIP() << "libtorch not available";
#endif
}

}  // namespace fuzzformer

