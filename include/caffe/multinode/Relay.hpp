#ifndef CAFFE_PARAMRELAY_HPP_
#define CAFFE_PARAMRELAY_HPP_

#include <string>
#include "caffe/internode/configuration.hpp"
#include "caffe/solver.hpp"

namespace caffe {

template <typename Dtype>
class ParamRelay {
  class Impl;
  shared_ptr<Impl> impl;
 public:
  ParamRelay(shared_ptr<Solver<Dtype> > solver,
             string bind_address,
             string param_server_address,
             int ignored_threads);
  void run();
};
}  // namespace caffe

#endif  // CAFFE_PARAMRELAY_HPP_

