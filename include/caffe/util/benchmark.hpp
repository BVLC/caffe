#ifndef CAFFE_UTIL_BENCHMARK_H_
#define CAFFE_UTIL_BENCHMARK_H_

#include <boost/date_time/posix_time/posix_time.hpp>

#include "caffe/util/device_alternate.hpp"

namespace caffe {

class Timer {
 public:
  Timer();
  virtual ~Timer();
  void Start();
  void Stop();
  float MilliSeconds();
  float Seconds();

  inline bool initted() { return initted_; }
  inline bool running() { return running_; }
  inline bool has_run_at_least_once() { return has_run_at_least_once_; }

 protected:
  void Init();

  bool initted_;
  bool running_;
  bool has_run_at_least_once_;
#ifndef CPU_ONLY
  cudaEvent_t start_gpu_;
  cudaEvent_t stop_gpu_;
#endif
  boost::posix_time::ptime start_cpu_;
  boost::posix_time::ptime stop_cpu_;
  float elapsed_milliseconds_;
};

}  // namespace caffe

#endif   // CAFFE_UTIL_BENCHMARK_H_
