#include <gtest/gtest.h>

#include <chrono>
#include <sstream>
#include <thread>

#include "fuzzformer/logger.h"
#include "fuzzformer/tensorUtils.h"
#include "fuzzformer/timer.h"

namespace fuzzformer {

TEST(TimerTest, MeasuresElapsedTime) {
  Timer timer;
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  const auto elapsed = timer.elapsed();
  EXPECT_GE(elapsed.count(), 0.0);
}

TEST(LoggerTest, RespectsSeverityThreshold) {
  auto& logger = Logger::instance();
  std::ostringstream buffer;
  logger.set_stream(&buffer);
  logger.set_level(LogLevel::kWarn);

  log(LogLevel::kInfo, "suppressed");
  EXPECT_TRUE(buffer.str().empty());

  log(LogLevel::kError, "visible");
  ASSERT_FALSE(buffer.str().empty());
  EXPECT_NE(buffer.str().find("visible"), std::string::npos);
}

TEST(TensorUtilsTest, RejectsInvalidAttentionDims) {
  EXPECT_THROW(tensor::validate_attention_dims(0, 1, 1, 1), std::invalid_argument);
  EXPECT_THROW(tensor::validate_attention_dims(1, -1, 1, 1), std::invalid_argument);
  EXPECT_THROW(tensor::validate_attention_dims(1, 1, 0, 1), std::invalid_argument);
  EXPECT_THROW(tensor::validate_attention_dims(1, 1, 1, 0), std::invalid_argument);
  EXPECT_NO_THROW(tensor::validate_attention_dims(1, 1, 1, 1));
}

}  // namespace fuzzformer

