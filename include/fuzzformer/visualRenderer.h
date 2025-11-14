#pragma once

#include <vector>

#ifdef FUZZFORMER_HAS_TORCH
#include <torch/torch.h>
#else
#include "fuzzformer/torchStub.h"
#endif

namespace fuzzformer {

class VisualRenderer {
 public:
  VisualRenderer();
  ~VisualRenderer();

  void initialize();
  void shutdown();

  void render_attention(const torch::Tensor& attention);

  static void render_attention_heatmap(const std::vector<std::vector<float>>& attention_matrix);
  static void render_attention_1d(const std::vector<float>& attention_weights);

 private:
  bool initialized_;
};

}  // namespace fuzzformer

