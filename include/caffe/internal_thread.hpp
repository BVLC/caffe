#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }

namespace caffe {

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread();
  virtual ~InternalThread();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   * Will not return until the internal thread has exited.
   */
  bool StartInternalThread();

  /** Will not return until the internal thread has exited. */
  bool WaitForInternalThreadToExit();

  bool is_started() const;

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry() {}

 private:
  void entry();

  shared_ptr<boost::thread> thread_;
  int device_;
  Caffe::Brew mode_;
  int rand_seed_;
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
