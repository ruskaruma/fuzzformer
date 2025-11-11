#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace torch {

struct Device {
  constexpr bool is_cuda() const noexcept { return false; }
};

struct Tensor {
  Device device() const noexcept { return {}; }
  Tensor options() const { return {}; }
};

inline Tensor ones(const std::vector<std::size_t>&, const Tensor&) { return {}; }
inline Tensor zeros(const std::vector<std::size_t>&, const Tensor&) { return {}; }

namespace nn {

class Module {
 public:
  virtual ~Module() = default;

  template <typename ModuleType>
  ModuleType register_module(const std::string&, ModuleType module) {
    return module;
  }
};

class LinearOptions {
 public:
  LinearOptions(std::size_t /*in_features*/, std::size_t /*out_features*/) {}
  LinearOptions& bias(bool /*value*/) { return *this; }
};

class Linear : public Module {
 public:
  Linear() = default;
  explicit Linear(const LinearOptions& /*options*/) {}
  Tensor operator()(const Tensor& input) const { return input; }
};

}  // namespace nn

}  // namespace torch

#ifndef TORCH_CHECK
#define TORCH_CHECK(condition, message) \
  do {                                  \
    (void)(condition);                  \
  } while (0)
#endif

#ifndef TORCH_MODULE
#define TORCH_MODULE(Name) using Name = Name##Impl;
#endif

