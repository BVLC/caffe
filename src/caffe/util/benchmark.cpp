#include <boost/date_time/posix_time/posix_time.hpp>

#include "caffe/common.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

Timer::Timer()
    : initted_(false),
      running_(false),
      has_run_at_least_once_(false) {
  Init();
}

Timer::~Timer() {
  if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
    CUDA_CHECK(cudaEventDestroy(start_gpu_));
    CUDA_CHECK(cudaEventDestroy(stop_gpu_));
#else
    NO_GPU;
#endif
  }
}

void Timer::Start() {
  if (!running()) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      CUDA_CHECK(cudaEventRecord(start_gpu_, 0));
#else
      NO_GPU;
#endif
    } else {
      start_cpu_ = boost::posix_time::microsec_clock::local_time();
    }
    running_ = true;
    has_run_at_least_once_ = true;
  }
}

void Timer::Stop() {
  if (running()) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      CUDA_CHECK(cudaEventRecord(stop_gpu_, 0));
      CUDA_CHECK(cudaEventSynchronize(stop_gpu_));
#else
      NO_GPU;
#endif
    } else {
      stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    }
    running_ = false;
  }
}

float Timer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_milliseconds_, start_gpu_,
                                    stop_gpu_));
#else
      NO_GPU;
#endif
  } else {
    elapsed_milliseconds_ = (stop_cpu_ - start_cpu_).total_milliseconds();
  }
  return elapsed_milliseconds_;
}

float Timer::Seconds() {
  return MilliSeconds() / 1000.;
}

void Timer::Init() {
  if (!initted()) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      CUDA_CHECK(cudaEventCreate(&start_gpu_));
      CUDA_CHECK(cudaEventCreate(&stop_gpu_));
#else
      NO_GPU;
#endif
    }
    initted_ = true;
  }
}

}  // namespace caffe
