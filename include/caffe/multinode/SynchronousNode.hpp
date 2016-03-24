#ifndef CAFFE_SYNCHRONOUSNODE_HPP_
#define CAFFE_SYNCHRONOUSNODE_HPP_

#include <string>
#include "caffe/solver.hpp"

namespace caffe {

template <typename Dtype>
class SynchronousNode {
  class Impl;
  shared_ptr<Impl> impl;
 public:
  SynchronousNode(shared_ptr<Solver<Dtype> >, int num_of_threads);
  void run();
};
}  // namespace caffe


#endif  // CAFFE_SYNCHRONOUSNODE_HPP_

