#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include <boost/thread.hpp>

#include "caffe/common.hpp"

namespace caffe {

/**
 * Virutal class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virutal function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread() : thread_(NULL) {}
  virtual ~InternalThread();

  /** Returns true if the thread was successfully started. **/
  bool StartInternalThread();

  /** Will not return until the internal thread has exited. */
  bool WaitForInternalThreadToExit();

  bool is_started() const { return thread_ != NULL && thread_->joinable(); }

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry() {}

  boost::thread* thread_;
};

}  // namespace caffe

#endif
