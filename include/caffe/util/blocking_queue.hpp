<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#ifndef CAFFE_UTIL_BLOCKING_QUEUE_HPP_
#define CAFFE_UTIL_BLOCKING_QUEUE_HPP_

#include <queue>
#include <string>
=======
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
#ifndef CAFFE_UTIL_BLOCKING_QUEUE_H_
#define CAFFE_UTIL_BLOCKING_QUEUE_H_

#include <queue>
#include "boost/thread.hpp"
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel

namespace caffe {

template<typename T>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
class BlockingQueue {
 public:
  explicit BlockingQueue();

  void push(const T& t);

  bool try_pop(T* t);

  // This logs a message if the threads needs to be blocked
  // useful for detecting e.g. when data feeding is too slow
  T pop(const string& log_on_wait = "");

  bool try_peek(T* t);

  // Return element without removing it
  T peek();

  size_t size() const;

 protected:
  /**
   Move synchronization fields out instead of including boost/thread.hpp
   to avoid a boost/NVCC issues (#1009, #1010) on OSX. Also fails on
   Linux CUDA 7.0.18.
   */
  class sync;

  std::queue<T> queue_;
  shared_ptr<sync> sync_;

DISABLE_COPY_AND_ASSIGN(BlockingQueue);
=======
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
class blocking_queue {
public:
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
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
};

}  // namespace caffe

#endif
