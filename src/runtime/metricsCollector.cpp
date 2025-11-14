#include "fuzzformer/metricsCollector.h"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace fuzzformer {
namespace metrics {

MetricsCollector::MetricsCollector() {}

MetricsCollector::~MetricsCollector() {}

void MetricsCollector::start_collection(const std::string& kernel_name) {
  current_kernel_ = kernel_name;
  start_time_ = std::chrono::high_resolution_clock::now();
}

void MetricsCollector::stop_collection() {
  if (current_kernel_.empty()) {
    return;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_);

  KernelMetrics metrics;
  metrics.kernel_name = current_kernel_;
  metrics.duration_us = static_cast<double>(duration.count());

  metrics_[current_kernel_] = metrics;
  current_kernel_.clear();
}

KernelMetrics MetricsCollector::get_metrics(const std::string& kernel_name) const {
  auto it = metrics_.find(kernel_name);
  if (it != metrics_.end()) {
    return it->second;
  }
  return KernelMetrics{};
}

void MetricsCollector::render_throughput_card(const KernelMetrics& metrics) const {
  std::cout << "\n";
  std::cout << "─────────────────────────────────────────────────────\n";
  std::cout << "FuzzFormer Kernel: " << metrics.kernel_name << "\n";
  std::cout << "─────────────────────────────────────────────────────\n";
  std::cout << std::fixed << std::setprecision(1);
  std::cout << "Duration             : " << metrics.duration_us << " μs\n";
  if (metrics.dram_throughput_percent > 0.0) {
    std::cout << "DRAM Throughput      : " << metrics.dram_throughput_percent << "%\n";
  }
  if (metrics.l2_throughput_percent > 0.0) {
    std::cout << "L2 Cache Throughput  : " << metrics.l2_throughput_percent << "%\n";
  }
  if (metrics.sm_utilization_percent > 0.0) {
    std::cout << "SM Compute Util      : " << metrics.sm_utilization_percent << "%\n";
  }
  if (metrics.elapsed_cycles > 0) {
    std::cout << "Elapsed Cycles       : " << metrics.elapsed_cycles << "\n";
  }
  std::cout << "─────────────────────────────────────────────────────\n";
  std::cout << "\n";
}

}  // namespace metrics
}  // namespace fuzzformer

