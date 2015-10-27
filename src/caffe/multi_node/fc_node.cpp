

#include "caffe/multi_node/fc_node.hpp"


namespace caffe {

template <typename Dtype>
int FcNode<Dtype>::SetUpPoll()
{
  CHECK(this->poll_items_ != NULL);

  this->poll_items_[back_sock_index_].socket = sock_back_->GetSock();
  this->poll_items_[back_sock_index_].events = ZMQ_POLLIN;
  this->poll_items_[back_sock_index_].fd = 0;
  this->poll_items_[back_sock_index_].revents = 0;
  
  return MsgHub<Dtype>::SetUpPoll();
}

template <typename Dtype>
int FcNode<Dtype>::Init()
{
  const vector<string>& next = NodeEnv::Instance()->ClientAddrs();
  
  for (int i = 0; i < this->nworkers_; i++) {
    if (next.size() > 0) {
      this->threads_[i].reset(new FcThread<Dtype>());
    } else {
      this->threads_[i].reset(new FcEndThread<Dtype>());
    }
  }
  
  //the last slot for param thread
  this->threads_[param_thread_index_].reset(new FcParamThread<Dtype>());
  
  //wait for the downstream nodes to connect
  for (int i = 0; i < next.size(); i++) {
    shared_ptr<Msg> m = sock_back_->RecvMsg(true);

    CHECK (m->type() == PING);
    
    string node((char *)m->ZmsgData(0), m->ZmsgSize(0));
    LOG(INFO) << "Accepted Connection from: " << node;
  }
  
  num_next_hops_ = next.size();
  
  const SolverParameter& param = NodeEnv::Instance()->SolverParam();
  //set up solvers
  SGDSolver<Dtype> *pfc0 = new SGDSolver<Dtype>(param);
  NodeEnv::Instance()->PushFreeSolver(pfc0);

  //init the threads
  return this->StartThreads();
}

template <typename Dtype>
int FcNode<Dtype>::RouteMsg()
{
  //Got messages from the work threads:
  for (int i = 0; i < this->nworkers_; i++) {
    //
    if (this->poll_items_[i].revents & ZMQ_POLLIN) {
      shared_ptr<Msg> m = this->sockp_arr_[i]->RecvMsg(true);
      
      //route the msg to root thread for updating parameter
      if (m->dst() == ROOT_THREAD_ID) {
        this->sockp_arr_[param_thread_index_]->SendMsg(m);
      } else {
        SendOutMsg(m);
      }
    }
  }
  
  //Paramter update thread
  if (this->poll_items_[param_thread_index_].revents & ZMQ_POLLIN) {
    shared_ptr<Msg> m = this->sockp_arr_[param_thread_index_]->RecvMsg(true);
    
    //the parameter thread shouldn't send out any message
    LOG(ERROR) << "Received unsupported message";
  }

  //only deal with the backward packets from rear REP socket
  if (this->poll_items_[back_sock_index_].revents & ZMQ_POLLIN) {
    shared_ptr<Msg> m = sock_back_->RecvMsg(true);
    
    LOG(INFO) << "back server received message";
    
    //forward packets in the back server usually the labels
    if (m->type() == FORWARD) {
      this->ProcessMsg(m);
    } else if (m->type() == BACKWARD) {
      this->ProcessMsg(m);
    } else {
      LOG(ERROR) << "unknown message: ";
      m->PrintHeader();
    }
  }
  
  return 0;
}


//send out the message which has been processed by the threads
template <typename Dtype>
int FcNode<Dtype>::SendOutMsg(shared_ptr<Msg> m)
{
  //broadcast address, send the message to hub sock to broadcast
  if (m->dst() < 0) {
    sock_pub_->SendMsg(m);

    return 0;
  }
  
  //looking up the route table
  unordered_map<int, shared_ptr<SkSock> >::iterator iter = node_to_sock_.find(m->dst());

  CHECK( iter != node_to_sock_.end() ) << "Cannot find routes for id: " << m->dst();

  iter->second->SendMsg(m);

  return 0;
}

template <typename Dtype>
int FcNode<Dtype>::InitRoute()
{
  const vector<string>& dealer_addrs = NodeEnv::Instance()->DealerAddrs();
  const vector<int>& dealer_ids = NodeEnv::Instance()->DealerIDs();
  
  for (int i = 0; i < dealer_addrs.size(); i++) {
    //connect ROUTER node
    shared_ptr<SkSock> dealer(new SkSock(ZMQ_DEALER));
    dealer->SetId(node_id_);
    dealer->Connect(dealer_addrs[i]);
    
    //Unique CHECK
    CHECK(node_to_sock_.find(dealer_ids[i]) == node_to_sock_.end());
    node_to_sock_[dealer_ids[i]] = dealer;
    
    //send notification to upstream node
    shared_ptr<Msg> m(new Msg());
    m->set_src(node_id_);
    m->set_type(PING);
    m->AppendData(this->node_ip_.data(), this->node_ip_.length());

    dealer->SendMsg(m);

    vec_dealer_.push_back(dealer);
  }

  return 0;
}

template <typename Dtype>
int FcClient<Dtype>::Init()
{
  //Connect Upstream ROUTER & PUB nodes
  const vector<string>& sub_addrs = NodeEnv::Instance()->SubAddrs();

  for (int i = 0; i < sub_addrs.size(); i++) {
    //connect PUB node
    shared_ptr<SkSock> sub(new SkSock(ZMQ_SUB));
    sub->Connect(sub_addrs[i]);
    zmq_setsockopt(sub->GetSock(), ZMQ_SUBSCRIBE, "", 0);
    
    vec_sub_sock_.push_back(sub);
  }


  this->InitRoute();
  FcNode<Dtype>::Init();

  LOG(INFO) << "FcClient successfully initialized.";
  
  return 0;
}

template <typename Dtype>
int FcClient<Dtype>::SetUpPoll()
{
  this->num_poll_items_ = this->nthreads_ + 1 + vec_sub_sock_.size();
  this->poll_items_ = new zmq_pollitem_t[this->num_poll_items_];

  //adding the subscribers to the polling items
  int poll_index = sub_sock_index_;
  for (int i = 0; i < vec_sub_sock_.size(); i++, poll_index++) {
    this->poll_items_[poll_index].socket = vec_sub_sock_[i]->GetSock();
    this->poll_items_[poll_index].events = ZMQ_POLLIN;
    this->poll_items_[poll_index].fd = 0;
    this->poll_items_[poll_index].revents = 0;
  }

  return FcNode<Dtype>::SetUpPoll();
}

template <typename Dtype>
int FcClient<Dtype>::RouteMsg()
{
  int poll_index = sub_sock_index_;
  for (int i = 0; i < vec_sub_sock_.size(); i++, poll_index++) {
    if (this->poll_items_[poll_index].revents & ZMQ_POLLIN) {
      shared_ptr<Msg> m = vec_sub_sock_[i]->RecvMsg(true);
      
      LOG(INFO) << "sub received msg id: " << m->msg_id();

      this->ProcessMsg(m);
    }
  }

  return FcNode<Dtype>::RouteMsg();
}

template <typename Dtype>
int FcGateway<Dtype>::Init()
{
  //connect to the bottom layer (usally the loss layer)
  const vector<string>& bottom_addrs = NodeEnv::Instance()->BottomAddrs();
  const vector<int>& bottom_ids = NodeEnv::Instance()->BottomIDs();

  for (int i = 0; i < bottom_addrs.size(); i++) {
    shared_ptr<SkSock> dealer(new SkSock(ZMQ_DEALER));
    dealer->SetId(this->node_id_);
    dealer->Connect(bottom_addrs[i]);
    //Unique CHECK
    CHECK(this->node_to_sock_.find(bottom_ids[i]) == this->node_to_sock_.end() );
    this->node_to_sock_[bottom_ids[i]] = dealer;

    bottom_socks_.push_back(dealer);
  }

  this->InitRoute();
  FcNode<Dtype>::Init();

  LOG(INFO) << "FcGateway successfully inited.";
  
  return 0;
}

template <typename Dtype>
int FcGateway<Dtype>::SetUpPoll()
{
  //backward sock for downstream nodes
  // + server sock for the conv clients
  this->num_poll_items_ = this->nthreads_ + 2;
  this->poll_items_ = new zmq_pollitem_t[this->num_poll_items_];

  this->poll_items_[server_sock_index_].socket = sock_server_->GetSock();
  this->poll_items_[server_sock_index_].events = ZMQ_POLLIN;
  this->poll_items_[server_sock_index_].fd = 0;
  this->poll_items_[server_sock_index_].revents = 0;
 
  return FcNode<Dtype>::SetUpPoll();
}

template <typename Dtype>
int FcGateway<Dtype>::RouteMsg()
{
  //process the packets comes from the convolution clients
  if (this->poll_items_[server_sock_index_].revents & ZMQ_POLLIN) {
    shared_ptr<Msg> m = sock_server_->RecvMsg(true);
    
    //adding a unique id for each incoming message
    msg_id_++;
    m->set_msg_id(msg_id_);
    
    this->ProcessMsg(m);
  }

  return FcNode<Dtype>::RouteMsg();
}

template <typename Dtype>
int FcGateway<Dtype>::SendOutMsg(shared_ptr<Msg> m)
{
  //directly send out the message to the conv clients
  if (m->type() == BACKWARD) {
    sock_server_->SendMsg(m);

    return 0;
  }

  return FcNode<Dtype>::SendOutMsg(m);
}

INSTANTIATE_CLASS(FcGateway);
INSTANTIATE_CLASS(FcNode);
INSTANTIATE_CLASS(FcClient);

} //end caffe

