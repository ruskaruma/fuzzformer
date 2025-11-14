#include <gtest/gtest.h>

#include <thread>

#include "fuzzformer/metricsCollector.h"

namespace fuzzformer {
namespace metrics {

TEST(MetricsCollectorTest, CollectsKernelMetrics) {
  MetricsCollector collector;

  collector.start_collection("test_kernel");
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  collector.stop_collection();

  auto metrics = collector.get_metrics("test_kernel");
  EXPECT_EQ(metrics.kernel_name, "test_kernel");
  EXPECT_GT(metrics.duration_us, 0.0);
}

TEST(MetricsCollectorTest, RendersThroughputCard) {
  MetricsCollector collector;
  KernelMetrics metrics;
  metrics.kernel_name = "fuzzyAttentionForward";
  metrics.duration_us = 179.5;
  metrics.dram_throughput_percent = 81.2;
  metrics.l2_throughput_percent = 52.9;
  metrics.sm_utilization_percent = 44.8;
  metrics.elapsed_cycles = 128921;

  EXPECT_NO_THROW(collector.render_throughput_card(metrics));
}

TEST(MetricsCollectorTest, HandlesMissingMetrics) {
  MetricsCollector collector;
  auto metrics = collector.get_metrics("nonexistent");
  EXPECT_TRUE(metrics.kernel_name.empty());
  EXPECT_EQ(metrics.duration_us, 0.0);
}

}  // namespace metrics
}  // namespace fuzzformer

