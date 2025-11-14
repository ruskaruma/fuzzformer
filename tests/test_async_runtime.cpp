#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "fuzzformer/asyncScheduler.h"
#include "fuzzformer/eventLoop.h"

namespace fuzzformer {
namespace runtime {

TEST(EventLoopTest, ExecutesScheduledTasks) {
  EventLoop loop;
  std::atomic<bool> task_executed{false};

  std::thread worker([&loop] { loop.run(); });

  loop.schedule([&task_executed] { task_executed = true; });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  loop.stop();
  worker.join();

  EXPECT_TRUE(task_executed.load());
}

TEST(AsyncSchedulerTest, SchedulesAndExecutesTasks) {
  AsyncScheduler scheduler;
  std::atomic<bool> task_executed{false};

  scheduler.schedule_async([&task_executed] { task_executed = true; });
  
  for (int i = 0; i < 20 && !task_executed.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  EXPECT_TRUE(task_executed.load());
}

TEST(AsyncSchedulerTest, HandlesMultipleTasks) {
  AsyncScheduler scheduler;
  std::atomic<int> counter{0};

  for (int i = 0; i < 5; ++i) {
    scheduler.schedule_async([&counter] { counter++; });
  }
  
  for (int i = 0; i < 30 && counter.load() < 5; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  EXPECT_EQ(counter.load(), 5);
}

}  // namespace runtime
}  // namespace fuzzformer

