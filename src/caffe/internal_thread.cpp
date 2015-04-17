#include <boost/thread.hpp>
#include "caffe/internal_thread.hpp"

namespace caffe {

InternalThread::~InternalThread() {
  StopInternalThread();
}

bool InternalThread::is_started() const {
  return thread_.get() != NULL && thread_->joinable();
}


bool InternalThread::StartInternalThread() {
  if (!StopInternalThread()) {
    return false;
  }
  must_stop_ = false;
  try {
    thread_.reset(
        new boost::thread(&InternalThread::InternalThreadEntry, this));
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
