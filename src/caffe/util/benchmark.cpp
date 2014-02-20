// Copyright 2014 kloud@github

#include <ctime>
#include <cuda_runtime.h>

#include "caffe/common.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

Timer::Timer() {
  if (Caffe::mode() == Caffe::GPU) {
    cudaEventCreate (&start_gpu_);
    cudaEventCreate (&stop_gpu_);
  }
}

Timer::~Timer() {
  if (Caffe::mode() == Caffe::GPU) {
    cudaEventDestroy (start_gpu_);
    cudaEventDestroy (stop_gpu_);
  }
}

void Timer::Start() {
  if (Caffe::mode() == Caffe::GPU) {
    cudaEventRecord(start_gpu_, 0);
  } else {
    start_cpu_ = clock();
  }
}

void Timer::Stop() {
  if (Caffe::mode() == Caffe::GPU) {
    cudaEventRecord(stop_gpu_, 0);
  } else {
    stop_cpu_ = clock();
  }
}

float Timer::ElapsedSeconds() {
  float elapsed;
  if (Caffe::mode() == Caffe::GPU) {
    cudaEventSynchronize(stop_gpu_);
    cudaEventElapsedTime(&elapsed, start_gpu_, stop_gpu_);
    elapsed /= 1000.;
  } else {
    elapsed = float(stop_cpu_ - start_cpu_) / CLOCKS_PER_SEC;
  }
  return elapsed;
}

}  // namespace caffe
