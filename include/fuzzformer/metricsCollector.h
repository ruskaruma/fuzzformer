#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <memory>

namespace fuzzformer {
namespace metrics {

struct KernelMetrics {
  std::string kernel_name;
  double duration_us;
  double dram_throughput_percent;
  double l2_throughput_percent;
  double sm_utilization_percent;
  uint64_t elapsed_cycles;
  uint64_t bytes_read;
  uint64_t bytes_written;
  double occupancy_percent;
};

class MetricsCollector {
 public:
  MetricsCollector();
  ~MetricsCollector();

  void start_collection(const std::string& kernel_name);
  void stop_collection();
  KernelMetrics get_metrics(const std::string& kernel_name) const;

  void render_throughput_card(const KernelMetrics& metrics) const;

  // CUPTI-specific methods
  bool initialize_cupti();
  void shutdown_cupti();
  bool is_cupti_available() const { return cupti_available_; }

 private:
  std::unordered_map<std::string, KernelMetrics> metrics_;
  std::string current_kernel_;
  std::chrono::high_resolution_clock::time_point start_time_;
  
  // CUPTI state
  bool cupti_available_;
  CUpti_SubscriberHandle cupti_context_;
  CUpti_EventGroup cupti_event_group_;
  CUpti_MetricGroup cupti_metric_group_;
  
  // CUPTI callback helpers
  void collect_cupti_metrics(const std::string& kernel_name);
  bool setup_cupti_metrics();
  void read_cupti_metrics(KernelMetrics& metrics);
};

}  // namespace metrics
}  // namespace fuzzformer

