#include <boost/thread.hpp>
#include <string>

#include "caffe/data_layers.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template<typename T>
class blocking_queue<T>::sync {
 public:
  mutable boost::mutex mutex_;
  boost::condition_variable condition_;
};

template<typename T>
blocking_queue<T>::blocking_queue()
    : sync_(new sync()),
      last_wait_log_(time(0)),
      pops_() {
}

template<typename T>
blocking_queue<T>::~blocking_queue() {
}

template<typename T>
void blocking_queue<T>::push(const T& t) {
  boost::mutex::scoped_lock lock(sync_.get()->mutex_);
  queue_.push(t);
  lock.unlock();
  sync_.get()->condition_.notify_one();
}

template<typename T>
bool blocking_queue<T>::empty() const {
  boost::mutex::scoped_lock lock(sync_.get()->mutex_);
  return queue_.empty();
}
template<typename T>
bool blocking_queue<T>::try_pop(T* t) {
  boost::mutex::scoped_lock lock(sync_.get()->mutex_);

  if (queue_.empty())
    return false;

  *t = queue_.front();
  queue_.pop();
  return true;
}

template<typename T>
T blocking_queue<T>::pop(const string& log_on_wait) {
  boost::mutex::scoped_lock lock(sync_.get()->mutex_);

  while (queue_.empty()) {
    if (!log_on_wait.empty()) {
      time_t now = time(0);
      if (now - last_wait_log_ > 5) {
        last_wait_log_ = now;
        LOG(INFO)<< log_on_wait;
      }
    }
    sync_.get()->condition_.wait(lock);
  }

  T t = queue_.front();
  queue_.pop();
  pops_++;
  return t;
}

template<typename T>
T blocking_queue<T>::peek() {
  boost::mutex::scoped_lock lock(sync_.get()->mutex_);

  while (queue_.empty())
    sync_.get()->condition_.wait(lock);

  return queue_.front();
}

template class blocking_queue<Batch<float>*>;
template class blocking_queue<Batch<double>*>;
template class blocking_queue<Datum*>;

}  // namespace caffe
