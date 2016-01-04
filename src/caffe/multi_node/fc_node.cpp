

#include "caffe/multi_node/fc_node.hpp"
#include "caffe/multi_node/param_helper.hpp"


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
  const vector<string>& next = NodeEnv::Instance()->bcast_addrs();
  
  for (int i = 0; i < this->nworkers_; i++) {
    // if it doesn't broadcast, it means it has loss layer
    if (next.size() > 0) {
      this->threads_[i].reset(new FcThread<Dtype>());
    } else {
      this->threads_[i].reset(new FcLossThread<Dtype>());
    }
  }
  
  // the last slot for param thread
  this->threads_[param_thread_index_].reset(new FcParamThread<Dtype>());
  
  // wait for the downstream nodes to connect
  for (int i = 0; i < next.size(); i++) {
    shared_ptr<Msg> m = sock_back_->RecvMsg(true);

    CHECK (m->type() == PING);
    
    string node((char *)m->ZmsgData(0), m->ZmsgSize(0));
    LOG(INFO) << "Accepted Connection from: " << node;
  }
  
  num_next_hops_ = next.size();
  
  const SolverParameter& param = NodeEnv::Instance()->SolverParam();
  // set up solvers
  SGDSolver<Dtype> *pfc0 = new SGDSolver<Dtype>(param);
  
  // init input blobs
  shared_ptr<Net<Dtype> > net = pfc0->net();
  for (int i = 0; i < net->num_inputs(); i++) {
    int blob_index = net->input_blob_indices()[i];
    const string& blob_name = net->blob_names()[blob_index];
    input_blob_name_map_[blob_name] = true;
  }

  // clear root net's diff
  net->ClearParamDiffs();
  ParamHelper<Dtype>::CopyParamDataFromMsg(net, NodeEnv::Instance()->model_server_msg());
  NodeEnv::Instance()->PushFreeSolver(pfc0);

  // init the threads
  return this->StartThreads();
}

template <typename Dtype>
void FcNode<Dtype>::PrepareInputData(shared_ptr<Msg> m)
{
  /// check we got what we need
  for (int i = 0; i < m->num_blobs(); i++) {
    const BlobInfo& blob_info = m->blob_info(i);
    const string& blob_name = blob_info.blob_name();

    CHECK(is_input_blob(blob_name)) << "fatal: unknown blob: " << blob_name;
  }
  
  if (num_inputs() > m->num_blobs()) {
    // merge inputs together
    int64_t msg_id = m->msg_id();
    unordered_map<int64_t, shared_ptr<Msg> >::iterator iter = id_to_msg_.find(msg_id);

    if (iter == id_to_msg_.end()) {
      id_to_msg_[msg_id] = m;
    } else {
      // move the blobs in m to the buffer
      iter->second->MergeMsg(m);
    }
    
    // double check the hashed msg
    iter = id_to_msg_.find(msg_id);
    CHECK(iter != id_to_msg_.end()) << "cannot find msg id: " << msg_id;
    m = iter->second;

    if (m->num_blobs() == num_inputs()) {
      // remove the message from buffer and send it to workers
      id_to_msg_.erase(iter);
      this->ScheduleMsg(m);
    } else {
      // wait for more messages
      return;
    }
    
  } else {
    // do nothing for we've got all the inputs
    this->ScheduleMsg(m);
  }
}


template <typename Dtype>
int FcNode<Dtype>::RouteMsg()
{
  // Got messages from the work threads:
  for (int i = 0; i < this->nworkers_; i++) {
    //
    if (this->poll_items_[i].revents & ZMQ_POLLIN) {
      shared_ptr<Msg> m = this->sockp_arr_[i]->RecvMsg(true);
      
      // route the msg to root thread for updating parameter
      if (m->dst() == ROOT_THREAD_ID) {
        this->Enqueue(param_thread_index_, m);
      } else {
        SendOutMsg(m);
      }
    }
  }
  
  // Paramter update thread
  if (this->poll_items_[param_thread_index_].revents & ZMQ_POLLIN) {
    shared_ptr<Msg> m = this->sockp_arr_[param_thread_index_]->RecvMsg(true);
    
    if (m->type() == PUT_PARAM || m->type() == TRAIN_ITER) {
      sock_back_->SendMsg(m);
    } else {
      LOG(ERROR) << "Received unsupported message";
    }
  }

  // only deal with the backward packets from rear REP socket
  if (this->poll_items_[back_sock_index_].revents & ZMQ_POLLIN) {
    shared_ptr<Msg> m = sock_back_->RecvMsg(true);
    
    // forward packets in the back server usually the labels
    if (m->type() == FORWARD) {
      this->PrepareInputData(m);
    } else if (m->type() == BACKWARD) {
      this->ScheduleMsg(m);
    } else if (m->type() == GET_PARAM) {
      this->Enqueue(param_thread_index_, m);
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
  const vector<string>& prev_addrs = NodeEnv::Instance()->prev_router_addrs();
  const vector<int>& prev_ids = NodeEnv::Instance()->prev_node_ids();
  
  for (int i = 0; i < prev_addrs.size(); i++) {
    //connect ROUTER node
    shared_ptr<SkSock> dealer(new SkSock(ZMQ_DEALER));
    dealer->SetId(node_id_);
    dealer->Connect(prev_addrs[i]);
    
    //Unique CHECK
    CHECK(node_to_sock_.find(prev_ids[i]) == node_to_sock_.end());
    node_to_sock_[prev_ids[i]] = dealer;
    
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
  const vector<string>& sub_addrs = NodeEnv::Instance()->sub_addrs();

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

      this->PrepareInputData(m);
    }
  }

  return FcNode<Dtype>::RouteMsg();
}

template <typename Dtype>
int FcGateway<Dtype>::Init()
{
  //some blobs need to be forwarded to other nodes (usally the loss layer)
  const vector<string>& fwrd_addrs = NodeEnv::Instance()->forward_addrs();
  const vector<int>& fwrd_ids = NodeEnv::Instance()->forward_ids();
  const vector<string>& fwrd_blobs = NodeEnv::Instance()->forward_blobs();
  
  CHECK_EQ(fwrd_addrs.size(), fwrd_ids.size());
  CHECK_EQ(fwrd_addrs.size(), fwrd_blobs.size());

  for (int i = 0; i < fwrd_addrs.size(); i++) {
    shared_ptr<SkSock> dealer(new SkSock(ZMQ_DEALER));
    dealer->SetId(this->node_id_);
    dealer->Connect(fwrd_addrs[i]);
    //Unique CHECK
    CHECK(this->node_to_sock_.find(fwrd_ids[i]) == this->node_to_sock_.end() );
    this->node_to_sock_[fwrd_ids[i]] = dealer;

    fwrd_socks_.push_back(dealer);

    const string& blob_name = fwrd_blobs[i];
    unordered_map<string, shared_ptr<vector<int> > >::iterator iter = 
        fwrd_blob_name_to_ids_.find(blob_name);
    if (iter == fwrd_blob_name_to_ids_.end()) {
      shared_ptr<vector<int> > pvec(new vector<int>);
      pvec->push_back(fwrd_ids[i]);
      fwrd_blob_name_to_ids_[blob_name] = pvec;
    } else {
      iter->second->push_back(fwrd_ids[i]);
    }
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
void FcGateway<Dtype>::DemuxMsg(shared_ptr<Msg> m)
{
  for (int i = 0; i < m->num_blobs(); i++) {
    const BlobInfo& blob_info = m->blob_info(i);
    const string& blob_name = blob_info.blob_name();
    
    unordered_map<string, shared_ptr<vector<int> > >::iterator iter = 
      fwrd_blob_name_to_ids_.find(blob_name);

    // passthrough the non-forwardable blobs
    if (iter == fwrd_blob_name_to_ids_.end()) {
      continue;
    }

    // 
    shared_ptr<Msg> f = m->ExtractMsg(blob_name);
    shared_ptr<vector<int> > pids = iter->second;
    for (int j = 0; j < pids->size(); j++) {
      f->set_dst(pids->at(j));
      this->SendOutMsg(f);
    }
  }
  
  if (m->num_blobs() > 0) {
    this->PrepareInputData(m);
  }
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
    
    this->DemuxMsg(m);
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

