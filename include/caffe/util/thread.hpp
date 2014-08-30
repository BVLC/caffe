#ifndef CAFFE_THREAD_CPP_HPP_
#define CAFFE_THREAD_CPP_HPP_

#include <boost/thread.hpp>
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"

namespace caffe {

template<typename Callable, class A1>
Thread::Thread(Callable func, A1 a1) {
  this->thread_ = new boost::thread(func, a1);
}

void Thread::join() {
  static_cast<boost::thread*>(this->thread_)->join();
}

bool Thread::joinable() {
  return static_cast<boost::thread*>(this->thread_)->joinable();
}

}  // namespace caffe

#endif
