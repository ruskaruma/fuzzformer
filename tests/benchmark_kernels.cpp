#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "fuzzformer/fuzzyAttention.h"
#include "fuzzformer/metricsCollector.h"
#include "fuzzformer/model.h"
#include "fuzzformer/modelConfig.h"
#include "fuzzformer/timer.h"

#ifdef FUZZFORMER_HAS_TORCH
#include <torch/torch.h>
#endif

namespace fuzzformer {

class KernelBenchmark : public ::testing::Test {
 protected:
  void SetUp() override {
#ifdef FUZZFORMER_HAS_TORCH
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available";
    }
#endif
  }
};

#ifdef FUZZFORMER_HAS_TORCH
TEST_F(KernelBenchmark, FuzzyAttentionForwardThroughput) {
  const int batch_size = 4;
  const int num_heads = 8;
  const int seq_len = 128;
  const int head_dim = 64;
  const int num_iterations = 100;

  auto q = torch::randn({batch_size, num_heads, seq_len, head_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto k = torch::randn({batch_size, num_heads, seq_len, head_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto v = torch::randn({batch_size, num_heads, seq_len, head_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto alpha = torch::ones({num_heads}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto beta = torch::zeros({num_heads}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  torch::cuda::synchronize();
  Timer timer;
  timer.start();

  for (int i = 0; i < num_iterations; ++i) {
    auto output = fuzzy_attention_forward(q, k, v, alpha, beta);
    torch::cuda::synchronize();
  }

  auto elapsed = timer.elapsed();
  double avg_time_us = elapsed.count() / num_iterations;
  double throughput_gflops = (2.0 * batch_size * num_heads * seq_len * seq_len * head_dim) / (avg_time_us * 1e3);

  std::cout << "\nFuzzy Attention Forward Benchmark:\n";
  std::cout << "  Batch: " << batch_size << ", Heads: " << num_heads
            << ", SeqLen: " << seq_len << ", HeadDim: " << head_dim << "\n";
  std::cout << "  Avg Time: " << avg_time_us << " μs\n";
  std::cout << "  Throughput: " << throughput_gflops << " GFLOPs\n";

  EXPECT_GT(throughput_gflops, 0.0);
}

TEST_F(KernelBenchmark, ModelInferenceLatency) {
  ModelConfig config;
  config.num_layers = 4;
  config.model_dim = 256;
  config.num_heads = 8;

  auto model = FuzzFormer(config);
  auto input = torch::randn({2, 32, static_cast<long>(config.model_dim)},
                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  const int num_iterations = 50;
  torch::cuda::synchronize();
  Timer timer;
  timer.start();

  for (int i = 0; i < num_iterations; ++i) {
    auto output = model->forward(input);
    torch::cuda::synchronize();
  }

  auto elapsed = timer.elapsed();
  double avg_time_ms = elapsed.count() / num_iterations / 1000.0;

  std::cout << "\nModel Inference Benchmark:\n";
  std::cout << "  Layers: " << config.num_layers << ", Dim: " << config.model_dim
            << ", Heads: " << config.num_heads << "\n";
  std::cout << "  Avg Latency: " << avg_time_ms << " ms\n";

  EXPECT_GT(avg_time_ms, 0.0);
}

TEST_F(KernelBenchmark, MemoryBandwidthEstimate) {
  const int batch_size = 2;
  const int num_heads = 4;
  const int seq_len = 512;
  const int head_dim = 128;
  const int num_iterations = 20;

  auto q = torch::randn({batch_size, num_heads, seq_len, head_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto k = torch::randn({batch_size, num_heads, seq_len, head_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto v = torch::randn({batch_size, num_heads, seq_len, head_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto alpha = torch::ones({num_heads}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto beta = torch::zeros({num_heads}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  size_t input_bytes = (q.numel() + k.numel() + v.numel() + alpha.numel() + beta.numel()) * sizeof(float);
  size_t output_bytes = q.numel() * sizeof(float);
  size_t total_bytes = (input_bytes + output_bytes) * num_iterations;

  torch::cuda::synchronize();
  Timer timer;
  timer.start();

  for (int i = 0; i < num_iterations; ++i) {
    auto output = fuzzy_attention_forward(q, k, v, alpha, beta);
    torch::cuda::synchronize();
  }

  auto elapsed = timer.elapsed();
  double elapsed_seconds = elapsed.count() / 1e6;
  double bandwidth_gbps = (total_bytes / elapsed_seconds) / 1e9;

  std::cout << "\nMemory Bandwidth Estimate:\n";
  std::cout << "  Total Data: " << (total_bytes / 1e6) << " MB\n";
  std::cout << "  Bandwidth: " << bandwidth_gbps << " GB/s\n";

  EXPECT_GT(bandwidth_gbps, 0.0);
}
#else
TEST_F(KernelBenchmark, Placeholder) {
  GTEST_SKIP() << "libtorch not available for benchmarking";
}
#endif

}  // namespace fuzzformer

