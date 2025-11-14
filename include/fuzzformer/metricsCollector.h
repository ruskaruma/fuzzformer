#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

namespace fuzzformer {
namespace metrics {

struct KernelMetrics {
  std::string kernel_name;
  double duration_us;
  double dram_throughput_percent;
  double l2_throughput_percent;
  double sm_utilization_percent;
  uint64_t elapsed_cycles;
};

class MetricsCollector {
 public:
  MetricsCollector();
  ~MetricsCollector();

  void start_collection(const std::string& kernel_name);
  void stop_collection();
  KernelMetrics get_metrics(const std::string& kernel_name) const;

  void render_throughput_card(const KernelMetrics& metrics) const;

 private:
  std::unordered_map<std::string, KernelMetrics> metrics_;
  std::string current_kernel_;
  std::chrono::high_resolution_clock::time_point start_time_;
};

}  // namespace metrics
}  // namespace fuzzformer

