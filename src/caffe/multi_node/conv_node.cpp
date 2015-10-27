

#include "caffe/multi_node/conv_node.hpp"
#include "caffe/multi_node/conv_thread.hpp"

namespace caffe {

template <typename Dtype>
int ConvClient<Dtype>::Init()
{
  //at leaset two threads
  CHECK_GE(this->nthreads_, 2);

  //connecting to the gateway of fully connected layer
  fc_client_->Connect(fc_gateway_addr_);
  ps_client_->Connect(ps_addr_);
  
  //using a solver to share weights among threads
  Caffe::set_root_solver(true);
  SGDSolver<Dtype> *root_solver = new SGDSolver<Dtype>( NodeEnv::Instance()->SolverParam() );
  
  //init parameters from parameter server
  shared_ptr<Msg> ps_msg(new Msg());
  ps_msg->set_type(GET_PARAM);
  ps_msg->set_dst(PS_ID);
  ps_msg->set_src(NodeEnv::Instance()->ID());

  const vector<Blob<Dtype>*>& net_params = root_solver->net()->learnable_params();
  for (int i = 0; i < net_params.size(); i++) {
    ps_msg->AppendData(net_params[i]->cpu_diff(), net_params[i]->count() * sizeof(Dtype));
  }
  ps_client_->SendMsg(ps_msg);
  
  shared_ptr<Msg> r = ps_client_->RecvMsg(true);
  for (int i = 0; i < net_params.size(); i++) {
    CHECK_EQ(net_params[i]->count() * sizeof(Dtype), r->ZmsgSize(i));

    memcpy(net_params[i]->mutable_cpu_data(), r->ZmsgData(i), r->ZmsgSize(i));
  }

  //push as root solver
  NodeEnv::Instance()->PushFreeSolver(root_solver);
  
  CHECK_GE(this->threads_.size(), this->nthreads_);
  for (int i = 0; i < this->nworkers_; i++) {
    this->threads_[i].reset(new ConvThread<Dtype>());
  }

  this->threads_[ps_thread_index_].reset(new ConvParamThread<Dtype>());

  return this->StartThreads();
}

template <typename Dtype>
int ConvClient<Dtype>::SetUpPoll()
{
  this->num_poll_items_ = this->nthreads_ + 2;
  this->poll_items_ = new zmq_pollitem_t[this->num_poll_items_];
  
  //1 socket to communicate fc_gateway
  this->poll_items_[fc_sock_index_].socket = fc_client_->GetSock();
  this->poll_items_[fc_sock_index_].events = ZMQ_POLLIN;
  this->poll_items_[fc_sock_index_].fd = 0;
  this->poll_items_[fc_sock_index_].revents = 0;

  //1 socket to communicate parameter server
  this->poll_items_[ps_sock_index_].socket = ps_client_->GetSock();
  this->poll_items_[ps_sock_index_].events = ZMQ_POLLIN;
  this->poll_items_[ps_sock_index_].fd = 0;
  this->poll_items_[ps_sock_index_].revents = 0;

  return MsgHub<Dtype>::SetUpPoll();
}

template <typename Dtype>
int ConvClient<Dtype>::RouteMsg()
{
  for (int i = 0; i < this->nworkers_; i++) {
    if (this->poll_items_[i].revents & ZMQ_POLLIN) {
      shared_ptr<Msg> m = this->sockp_arr_[i]->RecvMsg(true);
      
      if (m->dst() == ROOT_THREAD_ID) {
        this->sockp_arr_[ps_thread_index_]->SendMsg(m);
      } else if (m->dst() == PS_ID) {
        ps_client_->SendMsg(m);
      } else {
        fc_client_->SendMsg(m);
      }
    }
  }
  
  //from the parameter client thread
  if (this->poll_items_[ps_thread_index_].revents & ZMQ_POLLIN) {
    shared_ptr<Msg> m = this->sockp_arr_[ps_thread_index_]->RecvMsg(true);
    ps_client_->SendMsg(m);
  }

  //incoming packet from fc gateway
  if (this->poll_items_[fc_sock_index_].revents & ZMQ_POLLIN) {
    shared_ptr<Msg> m = fc_client_->RecvMsg(true);

    this->sockp_arr_[0]->SendMsg(m);
    //ProcessMsg(m);
  }
  
  //incoming packet from parameter server
  if (this->poll_items_[ps_sock_index_].revents & ZMQ_POLLIN) {
    shared_ptr<Msg> m = ps_client_->RecvMsg(true);
    
    //forwarding the message from PS to PS client
    this->sockp_arr_[ps_thread_index_]->SendMsg(m);
  }

  return 0;
}

INSTANTIATE_CLASS(ConvClient);

}


