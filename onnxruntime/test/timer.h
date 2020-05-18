#pragma once
#include <algorithm>
#include <chrono>
#include <vector>
#include <numeric>

namespace onnxruntime {

template <typename T>
class TimeList {
 public:
  void Clear() { laps_t_.clear(); }
  void Add(T t) { laps_t_.push_back(t); }
  T Last(size_t offset = 0) const {
    if (!Size(offset)) {
      return 0;
    }
    return laps_t_.back();
  }
  T Max(size_t offset = 0) const {
    if (!Size(offset)) {
      return 0;
    }
    return *std::max_element((laps_t_.begin() + offset), laps_t_.end());
  }
  T Min(size_t offset = 0) const {
    if (!Size(offset)) {
      return 0;
    }
    return *std::min_element((laps_t_.begin() + offset), laps_t_.end());
  }
  T Sum(size_t offset = 0) const {
    if (!Size(offset)) {
      return 0;
    }
    return std::accumulate((laps_t_.begin() + offset), laps_t_.end(), 0.0);
  }
  size_t Size(size_t offset = 0) const {
    size_t size = (laps_t_.size() <= offset) ? 0 : (laps_t_.size() - offset);
    return size;
  }
  T Avg(size_t offset = 0) const {
    if (!Size(offset)) {
      return 0;
    }
    return Sum(offset) / Size(offset);
  }
  const std::vector<T>& Raw() const { return laps_t_; }

 private:
  std::vector<T> laps_t_;
};

class Timer {
 public:
  Timer() = default;
  virtual ~Timer() = default;

  void Reset() { laps_t_.Clear(); }
  void Start() { t_start_ = std::chrono::system_clock::now(); }
  float Stop() {
    t_stop_ = std::chrono::system_clock::now();
    auto ts = std::chrono::duration_cast<std::chrono::microseconds>(t_stop_ -
        t_start_);
    float elapse_ms = 1000.f * static_cast<float>(ts.count()) *
        std::chrono::microseconds::period::num /
        std::chrono::microseconds::period::den;
    this->laps_t_.Add(elapse_ms);
    return elapse_ms;
  }
  float AvgLapTimeMs() const { return laps_t_.Avg(); }
  const TimeList<float>& LapTimes() const { return laps_t_; }

 protected:
  TimeList<float> laps_t_;

 private:
  std::chrono::time_point<std::chrono::system_clock> t_start_, t_stop_;
};

}  // namespace onnxruntime
