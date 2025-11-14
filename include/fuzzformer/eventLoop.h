#pragma once

#include <functional>
#include <memory>

namespace fuzzformer {
namespace runtime {

class EventLoop {
 public:
  EventLoop();
  ~EventLoop();

  EventLoop(const EventLoop&) = delete;
  EventLoop& operator=(const EventLoop&) = delete;

  void run();
  void stop();
  bool is_running() const;

  void schedule(std::function<void()> task);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace runtime
}  // namespace fuzzformer

