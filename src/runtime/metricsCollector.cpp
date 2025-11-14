#include "fuzzformer/metricsCollector.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <mutex>

#ifdef HAVE_CUPTI
#include <cuda.h>
#include <cupti.h>

#ifndef CUPTIAPI
#define CUPTIAPI
#endif

#define CUPTI_CALL(call)                                                      \
  do {                                                                         \
    CUptiResult _status = call;                                               \
    if (_status != CUPTI_SUCCESS) {                                           \
      const char* errstr;                                                     \
      cuptiGetResultString(_status, &errstr);                                 \
      std::cerr << "CUPTI error at " << __FILE__ << ":" << __LINE__           \
                << " - " << errstr << std::endl;                              \
    }                                                                          \
  } while (0)

static std::mutex cupti_mutex;
static MetricsCollector* g_collector = nullptr;
static CUcontext g_cuda_context = nullptr;

extern "C" void CUPTIAPI cupti_callback_wrapper(void* userdata,
                                                 CUpti_CallbackDomain domain,
                                                 CUpti_CallbackId cbid,
                                                 const void* cbdata) {
  if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) {
    return;
  }

  if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
      cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v3020) {
    const CUpti_CallbackData* callbackData =
        static_cast<const CUpti_CallbackData*>(cbdata);
    if (callbackData->callbackSite == CUPTI_API_ENTER) {
      // Kernel launch detected
      if (g_collector) {
        std::lock_guard<std::mutex> lock(cupti_mutex);
        // Store kernel name for metrics collection
      }
    }
  }
}
#endif

namespace fuzzformer {
namespace metrics {

MetricsCollector::MetricsCollector()
    : cupti_available_(false), cupti_context_(nullptr), 
      cupti_event_group_(nullptr), cupti_metric_group_(nullptr) {
#ifdef HAVE_CUPTI
  initialize_cupti();
#endif
}

MetricsCollector::~MetricsCollector() {
#ifdef HAVE_CUPTI
  shutdown_cupti();
#endif
}

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
  if (metrics.occupancy_percent > 0.0) {
    std::cout << "Occupancy            : " << metrics.occupancy_percent << "%\n";
  }
  if (metrics.elapsed_cycles > 0) {
    std::cout << "Elapsed Cycles       : " << metrics.elapsed_cycles << "\n";
  }
  if (metrics.bytes_read > 0 || metrics.bytes_written > 0) {
    std::cout << "Memory Traffic        : " << (metrics.bytes_read + metrics.bytes_written) / 1024.0 / 1024.0 << " MB\n";
  }
  std::cout << "─────────────────────────────────────────────────────\n";
  std::cout << "\n";
}

#ifdef HAVE_CUPTI
bool MetricsCollector::initialize_cupti() {
  CUptiResult status;
  CUdevice device = 0;
  
  // Initialize CUDA context if not already done
  if (g_cuda_context == nullptr) {
    CUresult cu_status = cuCtxGetCurrent(&g_cuda_context);
    if (cu_status != CUDA_SUCCESS && cu_status != CUDA_ERROR_INVALID_CONTEXT) {
      cupti_available_ = false;
      return false;
    }
  }

  CUpti_SubscriberHandle subscriber;
  status = cuptiSubscribe(&subscriber, cupti_callback_wrapper, this);
  cupti_context_ = static_cast<void*>(subscriber);
  if (status != CUPTI_SUCCESS) {
    cupti_available_ = false;
    return false;
  }

  status = cuptiEnableDomain(1, static_cast<CUpti_SubscriberHandle>(cupti_context_), 
                            CUPTI_CB_DOMAIN_RUNTIME_API);
  if (status != CUPTI_SUCCESS) {
    cuptiUnsubscribe(static_cast<CUpti_SubscriberHandle>(cupti_context_));
    cupti_context_ = nullptr;
    cupti_available_ = false;
    return false;
  }

  // Setup metric collection
  if (!setup_cupti_metrics()) {
    cuptiUnsubscribe(static_cast<CUpti_SubscriberHandle>(cupti_context_));
    cupti_context_ = nullptr;
    cupti_available_ = false;
    return false;
  }

  g_collector = this;
  cupti_available_ = true;
  return true;
}

bool MetricsCollector::setup_cupti_metrics() {
  CUptiResult status;
  CUdevice device = 0;
  
  // Get device
  CUresult cu_status = cuDeviceGet(&device, 0);
  if (cu_status != CUDA_SUCCESS) {
    return false;
  }

  // Create event group for memory throughput
  CUpti_EventGroup event_group;
  status = cuptiEventGroupCreate(static_cast<CUpti_SubscriberHandle>(cupti_context_), 
                                 device, &event_group);
  if (status != CUPTI_SUCCESS) {
    return false;
  }
  cupti_event_group_ = static_cast<void*>(event_group);

  // Add events for memory throughput
  // These are common CUPTI events - actual availability depends on GPU architecture
  const char* events[] = {
    "dram_read_bytes",
    "dram_write_bytes",
    "l2_read_bytes",
    "l2_write_bytes",
    "sm__cycles_elapsed.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active"
  };

  int events_added = 0;
  for (const char* event_name : events) {
    CUpti_EventID event_id;
    status = cuptiEventGetIdFromName(device, event_name, &event_id);
    if (status == CUPTI_SUCCESS) {
      CUptiResult add_status = cuptiEventGroupAddEvent(
          static_cast<CUpti_EventGroup>(cupti_event_group_), event_id);
      if (add_status == CUPTI_SUCCESS) {
        events_added++;
      }
    }
  }

  // If no events were added, cleanup and return false
  if (events_added == 0) {
    cuptiEventGroupDestroy(static_cast<CUpti_EventGroup>(cupti_event_group_));
    cupti_event_group_ = nullptr;
    return false;
  }

  return true;
}

void MetricsCollector::shutdown_cupti() {
  if (cupti_available_ && cupti_context_) {
    if (cupti_event_group_) {
      CUPTI_CALL(cuptiEventGroupDestroy(static_cast<CUpti_EventGroup>(cupti_event_group_)));
      cupti_event_group_ = nullptr;
    }
    CUPTI_CALL(cuptiUnsubscribe(static_cast<CUpti_SubscriberHandle>(cupti_context_)));
    cupti_context_ = nullptr;
    cupti_available_ = false;
    g_collector = nullptr;
  }
}

void MetricsCollector::collect_cupti_metrics(const std::string& kernel_name) {
  if (!cupti_available_) {
    return;
  }

  auto it = metrics_.find(kernel_name);
  if (it != metrics_.end()) {
    read_cupti_metrics(it->second);
  }
}

void MetricsCollector::read_cupti_metrics(KernelMetrics& metrics) {
  if (!cupti_available_ || !cupti_event_group_) {
    return;
  }

  CUptiResult status;
  CUdevice device = 0;
  CUresult cu_status = cuDeviceGet(&device, 0);
  if (cu_status != CUDA_SUCCESS) {
    return;
  }

  CUpti_EventGroup event_group = static_cast<CUpti_EventGroup>(cupti_event_group_);
  
  // Enable event group
  status = cuptiEventGroupEnable(event_group);
  if (status != CUPTI_SUCCESS) {
    return;
  }

  // Read event values
  size_t event_count = 0;
  status = cuptiEventGroupGetNumEvents(event_group, &event_count);
  if (status != CUPTI_SUCCESS || event_count == 0) {
    cuptiEventGroupDisable(event_group);
    return;
  }

  std::vector<CUpti_EventID> event_ids(event_count);
  std::vector<uint64_t> event_values(event_count);
  
  // Get event IDs first
  size_t event_id_size = event_count * sizeof(CUpti_EventID);
  status = cuptiEventGroupGetAttribute(event_group,
                                       CUPTI_EVENT_GROUP_ATTR_EVENT_IDS,
                                       &event_id_size, event_ids.data());
  if (status != CUPTI_SUCCESS) {
    cuptiEventGroupDisable(event_group);
    return;
  }

  // Read all events
  size_t value_size = event_count * sizeof(uint64_t);
  status = cuptiEventGroupReadAllEvents(event_group, 
                                        CUPTI_EVENT_READ_FLAG_NONE,
                                        value_size,
                                        event_values.data(),
                                        event_id_size,
                                        event_ids.data(),
                                        &event_count);
  
  cuptiEventGroupDisable(event_group);

  if (status != CUPTI_SUCCESS) {
    return;
  }

  // Parse event values and populate metrics
  uint64_t dram_read_bytes = 0;
  uint64_t dram_write_bytes = 0;
  uint64_t l2_read_bytes = 0;
  uint64_t l2_write_bytes = 0;
  uint64_t total_cycles = 0;
  double occupancy_value = 0.0;
  
  for (size_t i = 0; i < event_count; ++i) {
    CUpti_EventID event_id = event_ids[i];
    uint64_t value = event_values[i];
    
    // Get event name to identify it
    char event_name[256];
    size_t name_size = sizeof(event_name);
    status = cuptiEventGetAttribute(event_id, CUPTI_EVENT_ATTR_NAME,
                                    &name_size, event_name);
    if (status == CUPTI_SUCCESS) {
      std::string name(event_name);
      if (name.find("dram_read") != std::string::npos) {
        dram_read_bytes += value;
      } else if (name.find("dram_write") != std::string::npos) {
        dram_write_bytes += value;
      } else if (name.find("l2_read") != std::string::npos) {
        l2_read_bytes += value;
      } else if (name.find("l2_write") != std::string::npos) {
        l2_write_bytes += value;
      } else if (name.find("cycles") != std::string::npos && 
                 name.find("elapsed") != std::string::npos) {
        total_cycles = value;
      } else if (name.find("warps_active") != std::string::npos ||
                 name.find("occupancy") != std::string::npos) {
        // Occupancy is typically a percentage value
        occupancy_value = static_cast<double>(value);
      }
    }
  }

  // Store separated read/write bytes
  metrics.bytes_read = dram_read_bytes;
  metrics.bytes_written = dram_write_bytes;
  metrics.elapsed_cycles = total_cycles;
  metrics.occupancy_percent = occupancy_value;
  
  // Calculate throughput percentages using real CUDA device properties
  CUdevice device = 0;
  CUresult cu_status = cuDeviceGet(&device, 0);
  if (cu_status == CUDA_SUCCESS && total_cycles > 0) {
    // Get device memory clock rate and bus width for bandwidth calculation
    int memory_clock_khz = 0;
    int bus_width = 0;
    cuDeviceGetAttribute(&memory_clock_khz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device);
    cuDeviceGetAttribute(&bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device);
    
    if (memory_clock_khz > 0 && bus_width > 0) {
      // Calculate peak memory bandwidth: (memory_clock * 2) * (bus_width / 8) / 1e9 GB/s
      // DDR memory transfers data on both edges, hence * 2
      double peak_bandwidth_gbps = (static_cast<double>(memory_clock_khz) * 2.0 * 
                                   static_cast<double>(bus_width) / 8.0) / 1e6;
      
      // Get device clock rate for cycle-to-time conversion
      int clock_rate_khz = 0;
      cuDeviceGetAttribute(&clock_rate_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device);
      
      if (clock_rate_khz > 0) {
        // Convert cycles to seconds: cycles / (clock_rate * 1000)
        double elapsed_seconds = static_cast<double>(total_cycles) / 
                                (static_cast<double>(clock_rate_khz) * 1000.0);
        
        if (elapsed_seconds > 0.0) {
          uint64_t total_dram_bytes = dram_read_bytes + dram_write_bytes;
          double bytes_per_second = static_cast<double>(total_dram_bytes) / elapsed_seconds;
          metrics.dram_throughput_percent = std::min(100.0, 
            (bytes_per_second / (peak_bandwidth_gbps * 1e9)) * 100.0);
          
          // Calculate L2 throughput if we have L2 cache bandwidth info
          // L2 bandwidth is typically higher than DRAM
          double l2_peak_bandwidth_gbps = peak_bandwidth_gbps * 2.0; // Estimate
          uint64_t total_l2_bytes = l2_read_bytes + l2_write_bytes;
          if (total_l2_bytes > 0) {
            double l2_bytes_per_second = static_cast<double>(total_l2_bytes) / elapsed_seconds;
            metrics.l2_throughput_percent = std::min(100.0,
              (l2_bytes_per_second / (l2_peak_bandwidth_gbps * 1e9)) * 100.0);
          }
        }
      }
    }
  }
}
#else
bool MetricsCollector::initialize_cupti() {
  cupti_available_ = false;
  return false;
}

void MetricsCollector::shutdown_cupti() {
  cupti_available_ = false;
}

void MetricsCollector::collect_cupti_metrics(const std::string& kernel_name) {
  // CUPTI not available without CUDA
}
#endif

}  // namespace metrics
}  // namespace fuzzformer

