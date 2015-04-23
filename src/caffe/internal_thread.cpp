#include <boost/thread.hpp>
#include <exception>

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

InternalThread::InternalThread()
    : thread_(),
      device_(),
      mode_(),
      rand_seed_(),
      solver_count_(),
      solver_index_() {
}

InternalThread::~InternalThread() {
  StopInternalThread();
}

bool InternalThread::is_started() const {
  return thread_.get() != NULL && thread_->joinable();
}

bool InternalThread::must_stop() {
  return thread_.get() != NULL && thread_->interruption_requested();
}

void InternalThread::StartInternalThread() {
  StopInternalThread();

#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
  mode_ = Caffe::mode();
  rand_seed_ = caffe_rng_rand();
  solver_count_ = Caffe::solver_count();
  solver_index_ = Caffe::solver_index();

  try {
    thread_.reset(new boost::thread(&InternalThread::entry, this));
  } catch (boost::thread_interrupted&) {
  } catch (std::exception& e) {
    CHECK(false) << e.what();
  }
}

void InternalThread::entry() {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaSetDevice(device_));
#endif
  Caffe::set_mode(mode_);
  Caffe::set_random_seed(rand_seed_);
  Caffe::set_solver_count(solver_count_);
  Caffe::set_solver_index(solver_index_);

  InternalThreadEntry();
}

void InternalThread::StopInternalThread() {
  if (is_started()) {
    thread_->interrupt();
    try {
      thread_->join();
    } catch (boost::thread_interrupted&) {
    } catch (std::exception& e) {
      CHECK(false) << e.what();
    }
  }
}

}  // namespace caffe
