#include <boost/thread.hpp>
#include <string>

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * 总体为生产者-消费者模式
 * empty() 堆栈为空则返回真
 * pop() 移除栈顶元素
 * push()	在栈顶增加元素
 * size()	返回栈中元素数目
 * top()	返回栈顶元素
*/
//首先尝试锁住，然后将数据push到队列（queue_ 是std::queue<T> 类型的），然后unlock，条件变量通知
template<typename T>
class BlockingQueue<T>::sync {
 public:
  mutable boost::mutex mutex_;
  boost::condition_variable condition_;
};

template<typename T>
BlockingQueue<T>::BlockingQueue()
    : sync_(new sync()) {
}

template<typename T>
void BlockingQueue<T>::push(const T& t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);
  queue_.push(t);
  lock.unlock();
  sync_->condition_.notify_one();
}

template<typename T>
bool BlockingQueue<T>::try_pop(T* t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  queue_.pop();
  return true;
}

template<typename T>
T BlockingQueue<T>::pop(const string& log_on_wait) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  while (queue_.empty()) {
    if (!log_on_wait.empty()) {
      LOG_EVERY_N(INFO, 1000)<< log_on_wait;
    }
    sync_->condition_.wait(lock);
  }

  T t = queue_.front();
  queue_.pop();
  return t;
}
//判断队列首部是不是有数据
template<typename T>
bool BlockingQueue<T>::try_peek(T* t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  return true;
}
//peek函数取出队列首部的数据，同样也是使用的条件变量来实现同步
template<typename T>
T BlockingQueue<T>::peek() {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  while (queue_.empty()) {
    sync_->condition_.wait(lock); //如果队列一直为空则一直在等待
  }

  return queue_.front();
}

template<typename T>
size_t BlockingQueue<T>::size() const {
  boost::mutex::scoped_lock lock(sync_->mutex_);
  return queue_.size();
}

template class BlockingQueue<Batch<float>*>;
template class BlockingQueue<Batch<double>*>;

}  // namespace caffe
