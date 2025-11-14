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
  bool initialize_opengl(int width = 800, int height = 600);
  void render_opengl_frame();
  void shutdown_opengl();

  static void render_attention_heatmap(const std::vector<std::vector<float>>& attention_matrix);
  static void render_attention_1d(const std::vector<float>& attention_weights);

 private:
  bool initialized_;
  bool opengl_initialized_;
  void* opengl_context_;
};

}  // namespace fuzzformer

