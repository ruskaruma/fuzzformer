#pragma once

#include <cstddef>

namespace fuzzformer {

struct ModelConfig {
  std::size_t model_dim = 128;
  std::size_t num_heads = 4;
  std::size_t ff_dim = 256;
  float dropout = 0.0F;
};

}  // namespace fuzzformer

