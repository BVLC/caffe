<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
#include <boost/thread.hpp>
#include <exception>

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
#include "caffe/internal_thread.hpp"

#include "caffe/util/thread.hpp"
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge

namespace caffe {

InternalThread::~InternalThread() {
  StopInternalThread();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
}

bool InternalThread::is_started() const {
  return thread_ && thread_->joinable();
}

bool InternalThread::must_stop() {
  return thread_ && thread_->interruption_requested();
}

void InternalThread::StartInternalThread() {
  CHECK(!is_started()) << "Threads should persist and not be restarted.";

  int device = 0;
#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDevice(&device));
#endif
  Caffe::Brew mode = Caffe::mode();
  int rand_seed = caffe_rng_rand();
  int solver_count = Caffe::solver_count();
  bool root_solver = Caffe::root_solver();

<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
  if (thread_ != NULL) {
    delete thread_;
  }
}

bool InternalThread::StartInternalThread() {
  if (!StopInternalThread()) {
    return false;
  }
  must_stop_ = false;
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
  try {
    thread_.reset(new boost::thread(&InternalThread::entry, this, device, mode,
          rand_seed, solver_count, root_solver));
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
void InternalThread::entry(int device, Caffe::Brew mode, int rand_seed,
    int solver_count, bool root_solver) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaSetDevice(device));
#endif
  Caffe::set_mode(mode);
  Caffe::set_random_seed(rand_seed);
  Caffe::set_solver_count(solver_count);
  Caffe::set_root_solver(root_solver);

  InternalThreadEntry();
}

void InternalThread::StopInternalThread() {
<<<<<<< HEAD
<<<<<<< HEAD
=======
/** Will not return until the internal thread has exited. */
bool InternalThread::StopInternalThread() {
  must_stop_ = true;
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
  if (is_started()) {
    thread_->interrupt();
    try {
      thread_->join();
    } catch (boost::thread_interrupted&) {
    } catch (std::exception& e) {
      LOG(FATAL) << "Thread exception: " << e.what();
    }
  }
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
  if (thread_ != NULL) {
    delete thread_;
  }
}

bool InternalThread::StartInternalThread() {
  if (!StopInternalThread()) {
    return false;
  }
  must_stop_ = false;
  try {
    thread_ = new caffe::Thread
        (&InternalThread::InternalThreadEntry, this);
  } catch (...) {
    return false;
  }
  return true;
}

/** Will not return until the internal thread has exited. */
bool InternalThread::StopInternalThread() {
  must_stop_ = true;
  if (is_started()) {
    try {
      thread_->join();
    } catch (...) {
      return false;
    }
  }
  return true;
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
}

}  // namespace caffe
