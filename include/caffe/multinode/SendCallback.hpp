#ifndef CAFFE_MULTINODE_SENDCALLBACK_HPP_
#define CAFFE_MULTINODE_SENDCALLBACK_HPP_

#include <boost/shared_ptr.hpp>
#include <string>

namespace caffe {

struct SendCallback {
  shared_ptr<string> buffer;
  SendCallback() : buffer(new string()) {}
  SendCallback(const SendCallback& other) : buffer(other.buffer) {}

  void operator()(bool) const {}
};

}  // namespace caffe

#endif  // CAFFE_MULTINODE_SENDCALLBACK_HPP_

