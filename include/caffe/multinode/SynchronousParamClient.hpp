#ifndef CAFFE_SYNCHRONOUSPARAMCLIENT_HPP_
#define CAFFE_SYNCHRONOUSPARAMCLIENT_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/internode/communication.hpp"
#include "caffe/layer.hpp"
#include "caffe/MultiSolver.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"

namespace caffe {

template<typename Dtype>
struct SynchronousParamSyncingImpl;

template<typename Dtype>
class SynchronousParamClient : public MultiSolver<Dtype>::Callback {
 public:
  explicit SynchronousParamClient(shared_ptr<Solver<Dtype> > solver,
                                  string param_server_address,
                                  int num_of_threads);
  virtual ~SynchronousParamClient();

  void run();

 protected:
  void on_start();
  void on_start(int layer_id);
  void on_forward_finished(int layer_id);
  void on_gradients_ready();
  void on_backward_start(int layer_id);
  void on_gradients_ready(int layer_id);

  shared_ptr<MultiSolver<Dtype> >  solver_;
  shared_ptr<SynchronousParamSyncingImpl<Dtype> > sync;
};

}  // namespace caffe

#endif  // CAFFE_SYNCHRONOUSPARAMCLIENT_HPP_

