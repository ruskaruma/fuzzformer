#pragma once

#ifdef FUZZFORMER_HAS_TORCH
#include <torch/torch.h>
#else
#include "fuzzformer/torchStub.h"
#endif

namespace fuzzformer {

class VisualRenderer {
 public:
  void initialize();
  void render_attention(const torch::Tensor& attention);
  void shutdown();
};

}  // namespace fuzzformer

