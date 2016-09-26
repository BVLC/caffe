
#include "caffe/multi_node/ps_node.hpp"
#include "caffe/multi_node/param_helper.hpp"

namespace caffe {

template <typename Dtype>
ParamServer<Dtype>::ParamServer(int nthreads)
                    : MsgHub<Dtype>(nthreads, nthreads) {
}

template <typename Dtype>
int ParamServer<Dtype>::Init() {
  this->InitRoute();

  for (int i = 0; i < this->nthreads_; i++) {
    this->threads_[i].reset(new PSThread<Dtype>());
  }

  const SolverParameter& param = NodeEnv::Instance()->SolverParam();
  // set up solvers
  Caffe::set_root_solver(true);
  SGDSolver<Dtype> *root_solver = new SGDSolver<Dtype>(param);

  // init root net
  root_solver->net()->ClearParamDiffs();
  ParamHelper<Dtype>::CopyParamDataFromMsg(root_solver->net(),
                              NodeEnv::Instance()->model_server_msg());

  NodeEnv::Instance()->SetRootSolver(root_solver);

  return this->StartThreads();
}

template <typename Dtype>
int ParamServer<Dtype>::RouteMsg() {
  if (this->poll_items_[this->back_sock_index_].revents & ZMQ_POLLIN) {
    shared_ptr<Msg> m = this->sock_back_->RecvMsg(true);

    // only 1 work thread on parameter server
    this->Enqueue(0, m);
  }

  for (int i = 0; i < this->nthreads_; i++) {
    if (this->poll_items_[i].revents & ZMQ_POLLIN) {
      shared_ptr<Msg> m = this->sockp_arr_[i]->RecvMsg(true);

      if (m->type() == EXIT_TRAIN) {
        return -1;
      }
      if (m->dst() < 0) {
        this->sock_pub_->SendMsg(m);
      } else {
        this->sock_back_->SendMsg(m);
      }
    }
  }

  return 0;
}

INSTANTIATE_CLASS(ParamServer);

}  // end namespace caffe


