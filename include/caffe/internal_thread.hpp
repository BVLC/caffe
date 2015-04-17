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
 * by reimplementing the virutal function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread() : thread_(), must_stop_() {}
  virtual ~InternalThread();

  /** Returns true if the thread was successfully started. **/
  bool StartInternalThread();

  /** Will not return until the internal thread has exited. */
  bool StopInternalThread();

  bool is_started() const;

  bool must_stop() {
    return must_stop_;
  }

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry() {}

  shared_ptr<boost::thread> thread_;
  bool must_stop_;
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
