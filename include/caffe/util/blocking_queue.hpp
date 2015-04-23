#ifndef CAFFE_UTIL_BLOCKING_QUEUE_H_
#define CAFFE_UTIL_BLOCKING_QUEUE_H_

#include <queue>
#include <string>

#include "caffe/common.hpp"

namespace caffe {

template<typename T>
class blocking_queue {
 public:
  explicit blocking_queue();
  virtual ~blocking_queue();

  void push(const T& t);

  bool empty() const;

  bool try_pop(T* t);

  T pop(const string& log_on_wait = "");

  // Return element without removing it
  T peek();

  inline uint64_t pops() {
    return pops_;
  }

 protected:
  /**
   Move synchronization fields out instead of including boost/thread.hpp
   to avoid a boost/NVCC issues (#1009, #1010) on OSX. Also fails on
   Linux CUDA 7.0.18.
   */
  class sync;

  std::queue<T> queue_;
  shared_ptr<sync> sync_;
  time_t last_wait_log_;
  uint64_t pops_;

DISABLE_COPY_AND_ASSIGN(blocking_queue);
};

}  // namespace caffe

#endif
