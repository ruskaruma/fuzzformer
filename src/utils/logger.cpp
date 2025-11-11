#include "fuzzformer/logger.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fuzzformer {

namespace {

constexpr std::string_view level_to_string(LogLevel level) {
  switch (level) {
    case LogLevel::kTrace:
      return "TRACE";
    case LogLevel::kDebug:
      return "DEBUG";
    case LogLevel::kInfo:
      return "INFO";
    case LogLevel::kWarn:
      return "WARN";
    case LogLevel::kError:
      return "ERROR";
  }
  return "UNKNOWN";
}

std::string timestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &time);
#else
  localtime_r(&time, &tm);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tm, "%H:%M:%S");
  return oss.str();
}

}  // namespace

Logger& Logger::instance() {
  static Logger logger;
  return logger;
}

Logger::Logger() : stream_(&std::clog), level_(LogLevel::kInfo) {}

void Logger::set_level(LogLevel level) {
  std::lock_guard<std::mutex> lock(mutex_);
  level_ = level;
}

LogLevel Logger::level() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return level_;
}

bool Logger::should_log(LogLevel level) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<int>(level) >= static_cast<int>(level_);
}

void Logger::log(LogLevel level, std::string_view message) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (static_cast<int>(level) < static_cast<int>(level_)) {
    return;
  }
  if (stream_) {
    (*stream_) << "[" << timestamp() << "] "
               << level_to_string(level) << ": "
               << message << '\n';
  }
}

void Logger::set_stream(std::ostream* stream) {
  std::lock_guard<std::mutex> lock(mutex_);
  stream_ = stream;
}

void log(LogLevel level, std::string_view message) {
  Logger::instance().log(level, message);
}

}  // namespace fuzzformer

