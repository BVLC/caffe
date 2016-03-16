#ifndef CAFFE_SYNCHRONOUSPARAMSERVER_HPP_
#define CAFFE_SYNCHRONOUSPARAMSERVER_HPP_

#include <string>
#include "caffe/internode/configuration.hpp"
#include "caffe/solver.hpp"

namespace caffe {

template <typename Dtype>
class SynchronousParamServer {
  class Impl;
  shared_ptr<Impl> impl;
 public:
  SynchronousParamServer(shared_ptr<Solver<Dtype> >,
                         string bind_address,
                         string ignored_address,
                         int num_of_threads);
  void run();
};
}  // namespace caffe


#endif  // CAFFE_SYNCHRONOUSPARAMSERVER_HPP_

