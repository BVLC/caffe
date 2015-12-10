

#include "caffe/multi_node/conv_node.hpp"
#include "caffe/multi_node/conv_thread.hpp"

namespace caffe {

template <typename Dtype>
int ConvClient<Dtype>::Init()
{
  //at leaset two threads
  CHECK_GE(this->nthreads_, 2);
  
  //connecting client socks
  fc_client_->Connect(fc_gateway_addr_);
  for (int i = 0; i < ps_num_; i++) {
    ps_clients_[i]->Connect(ps_addrs_[i]);
    int ps_id = ps_ids_[i];
    ps_node_to_sock_[ps_id] = ps_clients_[i];
  }

  LOG(INFO) << "Initing conv client";

  //using a solver to share weights among threads
  Caffe::set_root_solver(true);
  SGDSolver<Dtype> *root_solver = new SGDSolver<Dtype>( NodeEnv::Instance()->SolverParam() );

  // parameter server will assign a clock when register
  vector<int> ps_clocks;
  ps_clocks.resize(ps_num_);
  
  // init parameters from parameter server
  for (int i = 0; i < ps_num_; i++) {
    shared_ptr<Msg> ps_msg(new Msg());
    ps_msg->set_type(REGISTER_NODE);
    ps_msg->set_dst(ps_ids_[i]);
    ps_msg->set_src(NodeEnv::Instance()->ID());

    // add a integer to avoid sending message without data
    ps_msg->AppendData(&i, sizeof(i));

    ps_clients_[i]->SendMsg(ps_msg);
    
    LOG(INFO) << "waiting for parameter server : " << ps_addrs_[i];
    shared_ptr<Msg> m = ps_clients_[i]->RecvMsg(true);
    LOG(INFO) << "got response.";
    
    // copy initial clock and parameter
    ps_clocks[i] = m->clock();
    LOG(INFO) << "got clock: " << ps_clocks[i];
    ParamHelper<Dtype>::CopyParamDataFromMsg(root_solver->net(), m);
  }

  LOG(INFO) << "parameters inited";
  
  // push as root solver
  NodeEnv::Instance()->PushFreeSolver(root_solver);
  
  CHECK_GE(this->threads_.size(), this->nthreads_);
  for (int i = 0; i < this->nworkers_; i++) {
    this->threads_[i].reset(new ConvThread<Dtype>(ps_clocks));
  }

  this->threads_[ps_thread_index_].reset(new ConvParamThread<Dtype>());

  return this->StartThreads();
}

template <typename Dtype>
int ConvClient<Dtype>::SetUpPoll()
{
  this->num_poll_items_ = this->nthreads_ + 1 + ps_num_;
  this->poll_items_ = new zmq_pollitem_t[this->num_poll_items_];
  
  // 1 socket to communicate fc_gateway
  this->poll_items_[fc_sock_index_].socket = fc_client_->GetSock();
  this->poll_items_[fc_sock_index_].events = ZMQ_POLLIN;
  this->poll_items_[fc_sock_index_].fd = 0;
  this->poll_items_[fc_sock_index_].revents = 0;

  // sockets to communicate parameter server
  for (int i = 0; i < ps_num_; i++) {
    this->poll_items_[ps_sock_index_ + i].socket = ps_clients_[i]->GetSock();
    this->poll_items_[ps_sock_index_ + i].events = ZMQ_POLLIN;
    this->poll_items_[ps_sock_index_ + i].fd = 0;
    this->poll_items_[ps_sock_index_ + i].revents = 0;
  }

  return MsgHub<Dtype>::SetUpPoll();
}

template <typename Dtype>
int ConvClient<Dtype>::RouteMsg()
{
  for (int i = 0; i < this->nworkers_; i++) {
    if (this->poll_items_[i].revents & ZMQ_POLLIN) {
      shared_ptr<Msg> m = this->sockp_arr_[i]->RecvMsg(true);
      
      if (m->dst() == ROOT_THREAD_ID) {
        this->Enqueue(ps_thread_index_, m);
      } else if (m->type() == PUT_GRADIENT) {
        unordered_map<int, shared_ptr<SkSock> >::iterator iter = ps_node_to_sock_.find(m->dst());
    
        CHECK(iter != ps_node_to_sock_.end()) 
            << "cannot find socket for PS id: " << m->dst();
        iter->second->SendMsg(m);
      } else {
        fc_client_->SendMsg(m);
      }
    }
  }
  
  //from the parameter client thread to PS server
  if (this->poll_items_[ps_thread_index_].revents & ZMQ_POLLIN) {
    shared_ptr<Msg> m = this->sockp_arr_[ps_thread_index_]->RecvMsg(true);
    unordered_map<int, shared_ptr<SkSock> >::iterator iter = ps_node_to_sock_.find(m->dst());
    
    CHECK(iter != ps_node_to_sock_.end()) 
        << "cannot find socket for PS id: " << m->dst();
    iter->second->SendMsg(m);
  }

  //incoming packet from fc gateway
  if (this->poll_items_[fc_sock_index_].revents & ZMQ_POLLIN) {
    shared_ptr<Msg> m = fc_client_->RecvMsg(true);
    
    this->ScheduleMsg(m);
  }
  
  //incoming packet from parameter server
  for (int i = 0; i < ps_num_; i++) {
    int ps_poll_index = ps_sock_index_ + i;

    if (this->poll_items_[ps_poll_index].revents & ZMQ_POLLIN) {
      shared_ptr<Msg> m = ps_clients_[i]->RecvMsg(true);
    
      // forwarding the message to Param thread
      // this->sockp_arr_[ps_thread_index_]->SendMsg(m);
      this->ScheduleMsg(m);
    }
  }

  return 0;
}

INSTANTIATE_CLASS(ConvClient);

}


