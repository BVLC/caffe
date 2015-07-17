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
      start_cpu_ = std::chrono::high_resolution_clock::now();
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
      stop_cpu_ = std::chrono::high_resolution_clock::now();
    }
    running_ = false;
  }
}


float Timer::MicroSeconds() {
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
    // Cuda only measure milliseconds
    elapsed_microseconds_ = elapsed_milliseconds_ * 1000;
#else
      NO_GPU;
#endif
  } else {
    elapsed_microseconds_ = std::chrono::duration_cast
        <std::chrono::milliseconds>(stop_cpu_ - start_cpu_).count();
  }
  return elapsed_microseconds_;
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
    elapsed_milliseconds_ = std::chrono::duration_cast
        <std::chrono::milliseconds>(stop_cpu_ - start_cpu_).count();
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

CPUTimer::CPUTimer() {
  this->initted_ = true;
  this->running_ = false;
  this->has_run_at_least_once_ = false;
}

void CPUTimer::Start() {
  if (!running()) {
    this->start_cpu_ = std::chrono::high_resolution_clock::now();
    this->running_ = true;
    this->has_run_at_least_once_ = true;
  }
}

void CPUTimer::Stop() {
  if (running()) {
    this->stop_cpu_ = std::chrono::high_resolution_clock::now();
    this->running_ = false;
  }
}

float CPUTimer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  this->elapsed_milliseconds_ = std::chrono::duration_cast
      <std::chrono::milliseconds>(stop_cpu_ - start_cpu_).count();
  return this->elapsed_milliseconds_;
}

float CPUTimer::MicroSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  this->elapsed_microseconds_ = std::chrono::duration_cast
      <std::chrono::milliseconds>(stop_cpu_ - start_cpu_).count();
  return this->elapsed_microseconds_;
}

}  // namespace caffe
