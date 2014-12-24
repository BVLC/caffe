#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"

namespace caffe {

/**
 * A minimal wrapper for boost::thread to force host compilation for boost
 * Defined in caffe/util/thread.hpp
 */
class Thread {
 public:
  template<typename Callable, class A1>
  Thread(Callable func, A1 a1);
  void join();
  bool joinable();
 private:
  void* thread_;
};

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virutal function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread() : thread_(NULL), must_stop_() {}
  virtual ~InternalThread();

  /** Returns true if the thread was successfully started. **/
  bool StartInternalThread();

  /** Will not return until the internal thread has exited. */
  bool StopInternalThread();

  bool is_started() const { return thread_ != NULL && thread_->joinable(); }

  bool must_stop() {
    return must_stop_;
  }

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry() {}

  caffe::Thread* thread_;
  bool must_stop_;
};

}  // namespace caffe

#endif
