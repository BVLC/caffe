#include <boost/thread.hpp>

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

InternalThread::InternalThread()
    : thread_(),
      device_(),
      mode_(),
      rand_seed_() {
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

#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
  mode_ = Caffe::mode();
  rand_seed_ = caffe_rng_rand();

  try {
    thread_.reset(new boost::thread(&InternalThread::entry, this));
  } catch (...) {
    return false;
  }
  return true;
}

void InternalThread::entry() {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaSetDevice(device_));
#endif
  Caffe::set_mode(mode_);
  Caffe::set_random_seed(rand_seed_);

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
