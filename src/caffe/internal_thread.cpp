#include <iostream>

#include "caffe/internal_thread.hpp"

namespace caffe {
using namespace std;

InternalThread::~InternalThread() {
  WaitForInternalThreadToExit();
  if (thread_ != NULL) {
    delete thread_;
  }
}

bool InternalThread::StartInternalThread() {
  try {
    thread_ = new boost::thread(&InternalThread::InternalThreadEntry, this);
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
