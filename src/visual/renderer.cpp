#include "fuzzformer/visualRenderer.h"

#include <iostream>

#ifdef FUZZFORMER_HAS_TORCH
#include <torch/torch.h>
#endif

namespace fuzzformer {

VisualRenderer::VisualRenderer() : initialized_(false) {}

VisualRenderer::~VisualRenderer() {
  shutdown();
}

void VisualRenderer::initialize() {
  initialized_ = true;
}

void VisualRenderer::shutdown() {
  initialized_ = false;
}

void VisualRenderer::render_attention(const torch::Tensor& attention) {
#ifdef FUZZFORMER_HAS_TORCH
  if (!initialized_) {
    return;
  }

  if (attention.dim() != 2) {
    std::cerr << "Warning: Expected 2D attention matrix, got " << attention.dim() << "D\n";
    return;
  }

  const auto height = static_cast<int>(attention.size(0));
  const auto width = static_cast<int>(attention.size(1));

  std::vector<std::vector<float>> matrix(height);
  auto cpu_attention = attention.cpu().contiguous();
  auto accessor = cpu_attention.accessor<float, 2>();

  for (int i = 0; i < height; ++i) {
    matrix[i].resize(width);
    for (int j = 0; j < width; ++j) {
      matrix[i][j] = accessor[i][j];
    }
  }

  render_attention_heatmap(matrix);
#else
  std::cout << "VisualRenderer: libtorch not available, skipping render\n";
#endif
}

}  // namespace fuzzformer

