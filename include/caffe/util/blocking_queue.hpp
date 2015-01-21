#ifndef CAFFE_UTIL_BLOCKING_QUEUE_H_
#define CAFFE_UTIL_BLOCKING_QUEUE_H_

#include <queue>
#include "boost/thread.hpp"

namespace caffe {

template<typename T>
class blocking_queue {
 public:
  blocking_queue()
      : last_wait_log_(time(0)),
        pops_() {
  }

  void push(const T& t) {
    boost::mutex::scoped_lock lock(mutex_);
    queue_.push(t);
    lock.unlock();
    condition_.notify_one();
  }

  bool empty() const {
    boost::mutex::scoped_lock lock(mutex_);
    return queue_.empty();
  }

  bool try_pop(T& t) {
    boost::mutex::scoped_lock lock(mutex_);

    if (queue_.empty())
      return false;

    t = queue_.front();
    queue_.pop();
    return true;
  }

  T pop(const string& log_on_wait = "") {
    boost::mutex::scoped_lock lock(mutex_);

    while (queue_.empty()) {
      if (!log_on_wait.empty()) {
        time_t now = time(0);
        if (now - last_wait_log_ > 5) {
          last_wait_log_ = now;
          LOG(INFO) << log_on_wait;
        }
      }
      condition_.wait(lock);
    }

    T t = queue_.front();
    queue_.pop();
    pops_++;
    return t;
  }

  // Return element without removing it
  T peek() {
    boost::mutex::scoped_lock lock(mutex_);

    while (queue_.empty())
      condition_.wait(lock);

    return queue_.front();
  }

  inline uint64_t pops() {
    return pops_;
  }

 private:
  std::queue<T> queue_;
  mutable boost::mutex mutex_;
  boost::condition_variable condition_;
  time_t last_wait_log_;
  uint64_t pops_;
};

}  // namespace caffe

#endif
