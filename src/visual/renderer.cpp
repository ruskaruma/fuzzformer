#include "fuzzformer/visualRenderer.h"

#include <iostream>

#ifdef FUZZFORMER_HAS_TORCH
#include <torch/torch.h>
#endif

namespace fuzzformer {

VisualRenderer::VisualRenderer() : initialized_(false), opengl_initialized_(false), opengl_context_(nullptr) {}

VisualRenderer::~VisualRenderer() {
  shutdown();
  shutdown_opengl();
}

void VisualRenderer::initialize() {
  initialized_ = true;
}

void VisualRenderer::shutdown() {
  initialized_ = false;
}

bool VisualRenderer::initialize_opengl(int width, int height) {
  opengl_initialized_ = false;
  opengl_context_ = nullptr;
  return false;
}

void VisualRenderer::render_opengl_frame() {
  if (!opengl_initialized_) {
    return;
  }
}

void VisualRenderer::shutdown_opengl() {
  if (opengl_initialized_) {
    opengl_context_ = nullptr;
    opengl_initialized_ = false;
  }
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

