#pragma once

#include <chrono>

namespace fuzzformer {

class Timer {
 public:
  using clock = std::chrono::steady_clock;

  Timer();

  void reset();

  std::chrono::duration<double> elapsed() const;

 private:
  clock::time_point start_;
};

}  // namespace fuzzformer

