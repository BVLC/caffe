
#include "caffe/multi_node/ps_node.hpp"
#include "caffe/multi_node/param_helper.hpp"

namespace caffe {

template <typename Dtype>
ParamServer<Dtype>::ParamServer(int nthreads)
                    : MsgHub<Dtype>(nthreads, nthreads) {
  ps_router_.reset(new SkServer());
  ps_bind_addr_ = NodeEnv::Instance()->router_addr();
  ps_router_->Bind(ps_bind_addr_);

  ps_sock_index_ = nthreads;
}

template <typename Dtype>
int ParamServer<Dtype>::Init() {
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

  NodeEnv::Instance()->PushFreeSolver(root_solver);

  return this->StartThreads();
}

template <typename Dtype>
int ParamServer<Dtype>::RouteMsg() {
  if (this->poll_items_[ps_sock_index_].revents & ZMQ_POLLIN) {
    shared_ptr<Msg> m = ps_router_->RecvMsg(true);

    // only 1 work thread on parameter server
    this->Enqueue(0, m);
  }

  for (int i = 0; i < this->nthreads_; i++) {
    if (this->poll_items_[i].revents & ZMQ_POLLIN) {
      shared_ptr<Msg> m = this->sockp_arr_[i]->RecvMsg(true);

      if (m->type() == EXIT_TRAIN) {
        return -1;
      }
      ps_router_->SendMsg(m);
    }
  }

  return 0;
}

template <typename Dtype>
int ParamServer<Dtype>::SetUpPoll() {
  this->num_poll_items_ = this->nthreads_ + 1;
  this->poll_items_ = new zmq_pollitem_t[this->num_poll_items_];

  this->poll_items_[ps_sock_index_].socket = ps_router_->GetSock();
  this->poll_items_[ps_sock_index_].events = ZMQ_POLLIN;
  this->poll_items_[ps_sock_index_].fd = 0;
  this->poll_items_[ps_sock_index_].revents = 0;

  return MsgHub<Dtype>::SetUpPoll();
}

INSTANTIATE_CLASS(ParamServer);

}  // end namespace caffe


