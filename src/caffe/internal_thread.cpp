#include "caffe/internal_thread.hpp"

#include "caffe/util/thread.hpp"

namespace caffe {

InternalThread::~InternalThread() {
  WaitForInternalThreadToExit();
  if (thread_ != NULL) {
    delete thread_;
  }
}

bool InternalThread::StartInternalThread() {
  if (!WaitForInternalThreadToExit()) {
    return false;
  }
  try {
    thread_ = new caffe::Thread
        (&InternalThread::InternalThreadEntry, this);
  } catch (...) {
    return false;
  }
  return true;
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
