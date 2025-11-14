#ifdef FUZZFORMER_HAS_TORCH
#include <torch/torch.h>
#endif

#include <chrono>
#include <iostream>
#include <thread>

#include "fuzzformer/asyncScheduler.h"
#include "fuzzformer/model.h"
#include "fuzzformer/modelConfig.h"

int main() {
  fuzzformer::runtime::AsyncScheduler scheduler;

#ifdef FUZZFORMER_HAS_TORCH
  scheduler.schedule_async([&] {
    fuzzformer::ModelConfig config;
    auto model = fuzzformer::FuzzFormer(config);
    auto input = torch::randn({1, 8, static_cast<long>(config.model_dim)},
                              torch::TensorOptions().dtype(torch::kFloat32));
    auto output = model->forward(input);
    std::cout << "Inference output shape: " << output.sizes() << '\n';
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
#else
  scheduler.schedule_async([] {
    std::cout << "FuzzFormer scaffold ready.\n";
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
#endif

  return 0;
}

