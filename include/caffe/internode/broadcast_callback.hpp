#ifndef CAFFE_INTERNODE_BROADCASTCALLBACK_HPP_
#define CAFFE_INTERNODE_BROADCASTCALLBACK_HPP_

#include "boost/shared_ptr.hpp"

namespace caffe {
namespace internode {

template <typename SendCallback>
class BroadcastCallback {
  struct Impl {
    bool result;
    SendCallback callback;

    explicit Impl(SendCallback callback)
      : result(true)
      , callback(callback) {
    }

    void update(bool single_result) {
      result = result & single_result;
    }

    ~Impl() {
      callback(result);
    }
  };

  boost::shared_ptr<Impl> shared_impl;

 public:
  explicit BroadcastCallback(SendCallback callback)
    : shared_impl(new Impl(callback)) {
  }

  void operator()(bool single_result, ...) {
    shared_impl->update(single_result);
  }
};

}  // namespace internode
}  // namespace caffe

#endif  // CAFFE_INTERNODE_BROADCASTCALLBACK_HPP_


