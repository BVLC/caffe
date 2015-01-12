#include "caffe/internal_thread.hpp"

#include "caffe/util/thread.hpp"

namespace caffe {

InternalThread::~InternalThread() {
  StopInternalThread();
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
    thread_->interrupt();
    try {
      thread_->join();
    } catch (...) {
      return false;
    }
  }
  return true;
}

}  // namespace caffe
