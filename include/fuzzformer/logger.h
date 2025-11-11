#pragma once

#include <mutex>
#include <ostream>
#include <string_view>

namespace fuzzformer {

enum class LogLevel {
  kTrace = 0,
  kDebug,
  kInfo,
  kWarn,
  kError,
};

class Logger {
 public:
  static Logger& instance();

  void set_level(LogLevel level);

  [[nodiscard]] LogLevel level() const;

  [[nodiscard]] bool should_log(LogLevel level) const;

  void log(LogLevel level, std::string_view message);

  void set_stream(std::ostream* stream);

 private:
  Logger();

  mutable std::mutex mutex_;
  std::ostream* stream_;
  LogLevel level_;
};

void log(LogLevel level, std::string_view message);

}  // namespace fuzzformer

