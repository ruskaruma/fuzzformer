#ifdef FUZZFORMER_HAS_TORCH
#include <torch/torch.h>
#endif

#include <iostream>

#include "fuzzformer/model.h"
#include "fuzzformer/modelConfig.h"

int main() {
#ifdef FUZZFORMER_HAS_TORCH
  fuzzformer::ModelConfig config;
  auto model = fuzzformer::FuzzFormer(config);
  auto input = torch::randn({1, 8, static_cast<long>(config.model_dim)},
                            torch::TensorOptions().dtype(torch::kFloat32));
  auto output = model->forward(input);
  std::cout << "Inference output shape: " << output.sizes() << '\n';
#else
  std::cout << "FuzzFormer scaffold ready.\n";
#endif
  return 0;
}

