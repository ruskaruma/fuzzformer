#include <gtest/gtest.h>

#include "fuzzformer/model.h"
#include "fuzzformer/modelConfig.h"

#ifdef FUZZFORMER_HAS_TORCH
#include <torch/torch.h>
#endif

namespace fuzzformer {

TEST(ModelInferenceTest, OutputMatchesInputShape) {
#ifdef FUZZFORMER_HAS_TORCH
  ModelConfig config;
  config.num_layers = 2;

  auto model = FuzzFormer(config);
  auto input = torch::randn({2, 8, static_cast<long>(config.model_dim)},
                            torch::TensorOptions().dtype(torch::kFloat32));
  auto output = model->forward(input);
  ASSERT_EQ(output.sizes(), input.sizes());
#else
  GTEST_SKIP() << "libtorch not available";
#endif
}

}  // namespace fuzzformer



