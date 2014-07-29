// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include <pthread.h>

namespace caffe {

/**
 * Virutal class encapsulate pthread for use in base class
 * The child class will acquire the ability to run a single pthread,
 * by reimplementing the virutal function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread() {}
  virtual ~InternalThread() {}

  /** Returns true if the thread was successfully started **/
  bool StartInternalThread() {
    return pthread_create(&_thread, NULL, InternalThreadEntryFunc, this);
  }

  /** Will not return until the internal thread has exited. */
  bool WaitForInternalThreadToExit() {
    return pthread_join(_thread, NULL);
  }

 protected:
  /* Implement this method in your subclass 
      with the code you want your thread to run. */
  virtual void InternalThreadEntry() = 0;

 private:
  static void * InternalThreadEntryFunc(void * This) {
    reinterpret_cast<InternalThread *>(This)->InternalThreadEntry();
    return NULL;
  }

  pthread_t _thread;
};

}  // namespace caffe

#endif
