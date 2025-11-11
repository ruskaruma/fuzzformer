#include "fuzzformer/timer.h"

namespace fuzzformer {

Timer::Timer() : start_(clock::now()) {}

void Timer::reset() {
  start_ = clock::now();
}

std::chrono::duration<double> Timer::elapsed() const {
  return clock::now() - start_;
}

}  // namespace fuzzformer

