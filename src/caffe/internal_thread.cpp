#include <boost/thread.hpp>

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

InternalThread::~InternalThread() {
  StopInternalThread();
}

InternalThread::~InternalThread() {
  WaitForInternalThreadToExit();
}

bool InternalThread::is_started() const {
  return thread_.get() != NULL && thread_->joinable();
}

bool InternalThread::StartInternalThread() {
  if (!WaitForInternalThreadToExit()) {
    return false;
  }

  int device = 0;
#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDevice(&device));
#endif
  Caffe::Brew mode = Caffe::mode();
  int rand_seed = caffe_rng_rand();

  try {
    thread_.reset(new boost::thread(&InternalThread::entry, this, device, mode,
          rand_seed));
  } catch (...) {
    return false;
  }
  return true;
}

void InternalThread::entry(int device, Caffe::Brew mode, int rand_seed) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaSetDevice(device));
#endif
  Caffe::set_mode(mode);
  Caffe::set_random_seed(rand_seed);

  InternalThreadEntry();
}

/** Will not return until the internal thread has exited. */
bool InternalThread::WaitForInternalThreadToExit() {
  if (is_started()) {
    try {
      thread_->join();
    } catch (...) {
      return false;
    }
  }
  return true;
}

}  // namespace caffe
