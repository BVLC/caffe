#ifndef CAFFE_UTIL_BLOCKING_QUEUE_H_
#define CAFFE_UTIL_BLOCKING_QUEUE_H_

#include <boost/thread.hpp>
#include <queue>

namespace caffe {

template<typename T>
class blocking_queue {
 public:
  void push(const T& t) {
    boost::mutex::scoped_lock lock(mutex_);
    queue_.push(t);
    lock.unlock();
    cond_push_.notify_one();
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
    return t;
  }

  // Return element without removing it
  T peek() {
    boost::mutex::scoped_lock lock(mutex_);

    while (queue_.empty())
      condition_.wait(lock);

    return queue_.front();
  }

 private:
  std::queue<T> queue_;
  mutable boost::mutex mutex_;
  boost::condition_variable condition_;
  time_t last_wait_log_;
};

}  // namespace caffe

#endif
