#ifndef CAFFE_SYNCHRONOUSPARAMCLIENT_HPP_
#define CAFFE_SYNCHRONOUSPARAMCLIENT_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/MultiSolver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/internode/communication.hpp"

namespace caffe {

template<typename Dtype>
struct SynchronousParamSyncingImpl;

template<typename Dtype>
class SynchronousParamClient : public MultiSolver<Dtype>::Callback {
 public:
  explicit SynchronousParamClient(shared_ptr<Solver<Dtype> > solver,
                                  string param_server_address);
  virtual ~SynchronousParamClient();

  void run();

 protected:
  void on_start();
  void on_start(int);
  void on_forward_finished(int);
  void on_gradients_ready();
  void on_backward_start(int);
  void on_gradients_ready(int);

  shared_ptr<MultiSolver<Dtype> >  solver_;
  shared_ptr<SynchronousParamSyncingImpl<Dtype> > sync;
  int iters;
};

}  // namespace caffe

#endif //CAFFE_SYNCHRONOUSPARAMCLIENT_HPP_
