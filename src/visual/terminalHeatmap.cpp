#include "fuzzformer/visualRenderer.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace fuzzformer {

namespace {
constexpr const char* kHeatmapChars = " .:-=+*#%@";
constexpr int kHeatmapLevels = 9;

void render_heatmap_row(const std::vector<float>& row, int width) {
  if (row.empty()) {
    return;
  }
  const float min_val = *std::min_element(row.begin(), row.end());
  const float max_val = *std::max_element(row.begin(), row.end());
  const float range = max_val - min_val;

  for (int i = 0; i < width && i < static_cast<int>(row.size()); ++i) {
    float normalized = range > 0.0f ? (row[i] - min_val) / range : 0.0f;
    int level = static_cast<int>(normalized * (kHeatmapLevels - 1));
    level = std::clamp(level, 0, kHeatmapLevels - 1);
    std::cout << kHeatmapChars[level];
  }
  std::cout << '\n';
}

}  // namespace

void VisualRenderer::render_attention_heatmap(const std::vector<std::vector<float>>& attention_matrix) {
  if (attention_matrix.empty() || attention_matrix[0].empty()) {
    return;
  }

  const int height = static_cast<int>(attention_matrix.size());
  const int width = static_cast<int>(attention_matrix[0].size());

  std::cout << "\nAttention Heatmap (" << height << "x" << width << ")\n";
  std::cout << std::string(width, '-') << '\n';

  for (const auto& row : attention_matrix) {
    render_heatmap_row(row, width);
  }

  std::cout << std::string(width, '-') << '\n';
  std::cout << "Scale: . (min) to @ (max)\n\n";
}

void VisualRenderer::render_attention_1d(const std::vector<float>& attention_weights) {
  if (attention_weights.empty()) {
    return;
  }

  const int width = static_cast<int>(attention_weights.size());
  std::cout << "\nAttention Weights (" << width << ")\n";
  std::cout << std::string(width, '-') << '\n';
  render_heatmap_row(attention_weights, width);
  std::cout << std::string(width, '-') << '\n';
}

}  // namespace fuzzformer

