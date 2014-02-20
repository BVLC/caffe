// Copyright 2014 kloud@github

#ifndef CAFFE_UTIL_BENCHMARK_H_
#define CAFFE_UTIL_BENCHMARK_H_

#include <cuda_runtime.h>

namespace caffe {

class Timer {
 public:
  Timer();
  virtual ~Timer();
  void Start();
  void Stop();
  float ElapsedSeconds();

 protected:
  cudaEvent_t start_gpu_;
  cudaEvent_t stop_gpu_;
  clock_t start_cpu_;
  clock_t stop_cpu_;
};

}  // namespace caffe

#endif   // CAFFE_UTIL_BENCHMARK_H_
