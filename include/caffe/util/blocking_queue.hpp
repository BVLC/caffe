#ifndef CAFFE_UTIL_BLOCKING_QUEUE_H_
#define CAFFE_UTIL_BLOCKING_QUEUE_H_

#include <boost/thread.hpp>
#include <queue>

namespace caffe {

template<typename T>
class blocking_queue {
 public:
  explicit blocking_queue() { }
  virtual ~blocking_queue() { }

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

  T pop() {
    T t = peek();
    boost::mutex::scoped_lock lock(mutex_);
    queue_.pop();
    return t;
  }

  // Return element without removing it
  T peek() {
    boost::mutex::scoped_lock lock(mutex_);
    while (queue_.empty()) {
      cond_push_.wait(lock);
    }
    return queue_.front();
  }

 private:
  std::queue<T> queue_;
  mutable boost::mutex mutex_;
  boost::condition_variable cond_push_;

  DISABLE_COPY_AND_ASSIGN(blocking_queue);
};

}  // namespace caffe

#endif
