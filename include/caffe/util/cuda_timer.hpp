#ifndef CAFFE_CUDA_TIMER_HPP_
#define CAFFE_CUDA_TIMER_HPP_

#include "caffe/common.hpp"

namespace caffe {

class CudaTimer {

protected:
  cudaEvent_t start;
  cudaEvent_t stop;
  bool on;

public:
  CudaTimer() : on(false) {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~CudaTimer() {
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }

  void Tic() {
    CUDA_CHECK(cudaEventRecord(start, 0));
    on = true;
  }

  float Toc() {
    float time;
    CHECK(on) << "Tic before Toc";
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(start));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    on = false;
    return time;
  }
};

}  // namespace caffe

#endif // CAFFE_CUDA_TIMER_HPP_
