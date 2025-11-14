#include "fuzzformer/eventLoop.h"

#include <uv.h>

#include <functional>
#include <memory>
#include <queue>
#include <mutex>

namespace fuzzformer {
namespace runtime {

namespace {
struct TaskData {
  std::function<void()> task;
};

void on_async_callback(uv_async_t* handle) {
  auto* tasks = static_cast<std::queue<TaskData>*>(handle->data);
  std::mutex* mutex = static_cast<std::mutex*>(handle->loop->data);

  std::queue<TaskData> local_tasks;
  {
    std::lock_guard<std::mutex> lock(*mutex);
    local_tasks.swap(*tasks);
  }

  while (!local_tasks.empty()) {
    auto task_data = std::move(local_tasks.front());
    local_tasks.pop();
    task_data.task();
  }
}

void on_walk_callback(uv_handle_t* handle, void* arg) {
  if (uv_is_closing(handle)) {
    return;
  }
  uv_close(handle, nullptr);
}
}  // namespace

class EventLoop::Impl {
 public:
  Impl() : stopped_(false) {
    loop_ = std::make_unique<uv_loop_t>();
    uv_loop_init(loop_.get());
    loop_->data = &mutex_;

    async_handle_ = std::make_unique<uv_async_t>();
    uv_async_init(loop_.get(), async_handle_.get(), on_async_callback);
    async_handle_->data = &tasks_;
  }

  ~Impl() {
    stop();
    if (loop_) {
      uv_walk(loop_.get(), on_walk_callback, nullptr);
      uv_run(loop_.get(), UV_RUN_DEFAULT);
      uv_loop_close(loop_.get());
    }
  }

  void run() {
    uv_run(loop_.get(), UV_RUN_DEFAULT);
  }

  void stop() {
    if (stopped_ || !loop_) {
      return;
    }
    stopped_ = true;
    uv_stop(loop_.get());
    if (async_handle_ && !uv_is_closing(reinterpret_cast<uv_handle_t*>(async_handle_.get()))) {
      uv_async_send(async_handle_.get());
      uv_close(reinterpret_cast<uv_handle_t*>(async_handle_.get()), [](uv_handle_t* handle) {
        // Handle closed
      });
    }
  }

  bool is_running() const {
    return loop_ && uv_loop_alive(loop_.get());
  }

  void schedule(std::function<void()> task) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.push({std::move(task)});
    }
    uv_async_send(async_handle_.get());
  }

 private:
  std::unique_ptr<uv_loop_t> loop_;
  std::unique_ptr<uv_async_t> async_handle_;
  std::queue<TaskData> tasks_;
  mutable std::mutex mutex_;
  bool stopped_;
};

EventLoop::EventLoop() : impl_(std::make_unique<Impl>()) {}

EventLoop::~EventLoop() {
  stop();
}

void EventLoop::run() {
  impl_->run();
}

void EventLoop::stop() {
  impl_->stop();
}

bool EventLoop::is_running() const {
  return impl_->is_running();
}

void EventLoop::schedule(std::function<void()> task) {
  impl_->schedule(std::move(task));
}

}  // namespace runtime
}  // namespace fuzzformer

