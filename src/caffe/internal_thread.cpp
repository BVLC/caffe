#include "caffe/internal_thread.hpp"

namespace caffe {

InternalThread::~InternalThread() {
  WaitForInternalThreadToExit();
}

bool InternalThread::StartInternalThread() {
  if (!WaitForInternalThreadToExit()) {
    return false;
  }
  try {
    thread_.reset(new boost::thread(&InternalThread::InternalThreadEntry, this));
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
