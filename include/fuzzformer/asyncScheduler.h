#pragma once

#include <coroutine>
#include <functional>
#include <memory>

namespace fuzzformer {
namespace runtime {

class AsyncScheduler {
 public:
  AsyncScheduler();
  ~AsyncScheduler();

  void schedule_async(std::function<void()> task);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace runtime
}  // namespace fuzzformer

