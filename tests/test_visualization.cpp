#include <gtest/gtest.h>

#include <vector>

#include "fuzzformer/visualRenderer.h"

namespace fuzzformer {

TEST(VisualRendererTest, RendersHeatmapFromMatrix) {
  std::vector<std::vector<float>> matrix = {
      {0.1f, 0.2f, 0.3f, 0.4f},
      {0.5f, 0.6f, 0.7f, 0.8f},
      {0.2f, 0.3f, 0.4f, 0.5f},
  };

  EXPECT_NO_THROW(VisualRenderer::render_attention_heatmap(matrix));
}

TEST(VisualRendererTest, Renders1DAttention) {
  std::vector<float> weights = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};

  EXPECT_NO_THROW(VisualRenderer::render_attention_1d(weights));
}

TEST(VisualRendererTest, HandlesEmptyInput) {
  std::vector<std::vector<float>> empty_matrix;
  std::vector<float> empty_weights;

  EXPECT_NO_THROW(VisualRenderer::render_attention_heatmap(empty_matrix));
  EXPECT_NO_THROW(VisualRenderer::render_attention_1d(empty_weights));
}

}  // namespace fuzzformer

