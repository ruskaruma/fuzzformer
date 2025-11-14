#include "fuzzformer/asyncScheduler.h"
#include "fuzzformer/eventLoop.h"

#include <thread>

namespace fuzzformer {
namespace runtime {

class AsyncScheduler::Impl {
 public:
  Impl() : event_loop_(std::make_unique<EventLoop>()) {
    worker_ = std::thread([this] { event_loop_->run(); });
  }

  ~Impl() {
    if (event_loop_) {
      event_loop_->stop();
    }
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  void schedule_async(std::function<void()> task) {
    event_loop_->schedule(std::move(task));
  }

 private:
  std::unique_ptr<EventLoop> event_loop_;
  std::thread worker_;
};

AsyncScheduler::AsyncScheduler() : impl_(std::make_unique<Impl>()) {}

AsyncScheduler::~AsyncScheduler() = default;

void AsyncScheduler::schedule_async(std::function<void()> task) {
  impl_->schedule_async(std::move(task));
}

}  // namespace runtime
}  // namespace fuzzformer

