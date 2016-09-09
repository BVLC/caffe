
#include <string>
#include <vector>

#include "caffe/multi_node/conv_node.hpp"
#include "caffe/multi_node/conv_thread.hpp"

namespace caffe {

template <typename Dtype>
int ConvClient<Dtype>::Init() {
  // at leaset two threads
  CHECK_GE(this->nthreads_, 2);

  LOG(INFO) << "Initing conv client";

  // connect client socks
  for (int i = 0; i < gateway_num_; i++) {
    fc_clients_[i]->Connect(fc_gateway_addrs_[i]);
    int gt_id = fc_gateway_ids_[i];
    node_to_sock_[gt_id] = fc_clients_[i];
  }

  for (int i = 0; i < ps_num_; i++) {
    ps_clients_[i]->Connect(ps_addrs_[i]);
    int ps_id = ps_ids_[i];
    node_to_sock_[ps_id] = ps_clients_[i];
  }

  for (int i = 0; i < fwd_socks_.size(); i++) {
    fwd_socks_[i]->Connect(fc_fwd_addrs_[i]);
    int fwd_id = fc_fwd_ids_[i];
    node_to_sock_[fwd_id] = fwd_socks_[i];
  }

  // create solver
  Caffe::set_root_solver(true);

  #ifdef USE_FULL_SOLVER
  SGDSolver<Dtype> *full_solver = new SGDSolver<Dtype>(
                                      NodeEnv::Instance()->SolverParam());
  #endif

  // change the batch size according to thread number
  SolverParameter *psolver_param = NodeEnv::Instance()->mutable_SolverParam();

  NetParameter *pnet_param = psolver_param->mutable_net_param();
  for (int i = 0; i < pnet_param->layer_size(); i++) {
    LayerParameter *player = pnet_param->mutable_layer(i);
    const string& layer_type = player->type();

    if (layer_type == "Data" || layer_type == "AsyncData") {
      int batch_size = player->data_param().batch_size();
      CHECK_EQ(batch_size % this->nworkers_, 0)
              << "batch size should be a multiple of threads";

      player->mutable_data_param()->set_batch_size(
                                    batch_size / this->nworkers_);
    }
  }

  SGDSolver<Dtype> *root_solver = new SGDSolver<Dtype>(*psolver_param);

  // parameter server will assign a clock when register
  vector<int> ps_clocks;
  ps_clocks.resize(ps_num_);

  // register node to parameter server
  for (int i = 0; i < ps_num_; i++) {
    shared_ptr<Msg> ps_msg(new Msg());
    ps_msg->set_type(REGISTER_NODE);
    ps_msg->set_dst(ps_ids_[i]);
    ps_msg->set_src(NodeEnv::Instance()->ID());

    // add a integer to avoid sending message without data
    ps_msg->AppendData(&i, sizeof(i));

    ps_clients_[i]->SendMsg(ps_msg);
  }

  for (int i = 0; i < ps_num_; i++) {
    LOG(INFO) << "waiting for parameter server : " << ps_addrs_[i];
    shared_ptr<Msg> m = ps_clients_[i]->RecvMsg(true);

    // copy initial clock and parameter
    ps_clocks[i] = m->clock();
    LOG(INFO) << "got clock: " << ps_clocks[i];
    ParamHelper<Dtype>::CopyParamDataFromMsg(root_solver->net(), m);
  }

  LOG(INFO) << "parameters inited";

  // push as root solver
  NodeEnv::Instance()->PushFreeSolver(root_solver);

  #ifdef USE_FULL_SOLVER
  const vector<Blob<Dtype>*>& root_params =
                              root_solver->net()->learnable_params();
  // full solver share parameters with root solver
  const vector<Blob<Dtype>*>& full_params =
                              full_solver->net()->learnable_params();
  for (int i = 0; i < root_params.size(); i++) {
    CHECK_EQ(full_params[i]->count(), root_params[i]->count());
    full_params[i]->ShareData(*root_params[i]);
  }

  ConvThread<Dtype>::InitFullSolver(full_solver);
  ConvThread<Dtype>::InitBarrier(this->nworkers_);
  #endif

  CHECK_GE(this->threads_.size(), this->nthreads_);
  for (int i = 0; i < this->nworkers_; i++) {
    this->threads_[i].reset(new ConvThread<Dtype>());
  }

  this->threads_[ps_thread_index_].reset(
                                   new ConvParamThread<Dtype>(ps_clocks));

  return this->StartThreads();
}

template <typename Dtype>
int ConvClient<Dtype>::SetUpPoll() {
  this->num_poll_items_ = this->nthreads_ + gateway_num_ + ps_num_;
  this->poll_items_ = new zmq_pollitem_t[this->num_poll_items_];

  // 1 socket to communicate fc_gateway
  for (int i = 0; i < gateway_num_; i++) {
    this->poll_items_[fc_sock_index_ + i].socket = fc_clients_[i]->GetSock();
    this->poll_items_[fc_sock_index_ + i].events = ZMQ_POLLIN;
    this->poll_items_[fc_sock_index_ + i].fd = 0;
    this->poll_items_[fc_sock_index_ + i].revents = 0;
  }

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
void ConvClient<Dtype>::SendOutMsg(shared_ptr<Msg> m) {
  // broadcast the message to gateways
  if (m->dst() < 0) {
    for (int i = 0; i < fc_clients_.size(); i++) {
      fc_clients_[i]->SendMsg(m);
    }
  } else if (m->dst() == ROOT_THREAD_ID) {
    this->Enqueue(ps_thread_index_, m);
  } else if (m->dst() == WORKER_BCAST) {
    // broadcast the message to all the workers
    for (int i = 0; i < this->nthreads_; i++) {
      if (i != ps_thread_index_) {
        this->Enqueue(i, m);
      }
    }
  } else {
    // look up the routing table
    unordered_map<int, shared_ptr<SkSock> >::iterator iter =
                                             node_to_sock_.find(m->dst());
    CHECK(iter != node_to_sock_.end())
          << "cannot find socket for id: " << m->dst();
    iter->second->SendMsg(m);
  }
}

template <typename Dtype>
int ConvClient<Dtype>::RouteMsg() {
  for (int i = 0; i < this->nworkers_; i++) {
    if (this->poll_items_[i].revents & ZMQ_POLLIN) {
      shared_ptr<Msg> m = this->sockp_arr_[i]->RecvMsg(true);

      SendOutMsg(m);
    }
  }

  bool need_exit = false;
  // from the parameter client thread to PS server
  if (this->poll_items_[ps_thread_index_].revents & ZMQ_POLLIN) {
    shared_ptr<Msg> m = this->sockp_arr_[ps_thread_index_]->RecvMsg(true);
    if (m->type() == EXIT_TRAIN) {
      need_exit = true;
    }
    // send backward messages to worker thread
    if (m->type() == BACKWARD) {
      this->Enqueue(m->dst(), m);
    } else {
      SendOutMsg(m);
    }
  }

  // incoming packet from fc gateway
  for (int i = 0; i < gateway_num_; i++) {
    int fc_poll_idx = fc_sock_index_ + i;
    if (this->poll_items_[fc_poll_idx].revents & ZMQ_POLLIN) {
      shared_ptr<Msg> m = fc_clients_[i]->RecvMsg(true);

      this->Enqueue(ps_thread_index_, m);
    }
  }

  // incoming packet from parameter server
  for (int i = 0; i < ps_num_; i++) {
    int ps_poll_index = ps_sock_index_ + i;

    if (this->poll_items_[ps_poll_index].revents & ZMQ_POLLIN) {
      shared_ptr<Msg> m = ps_clients_[i]->RecvMsg(true);

      // forwarding the message to Param thread
      this->sockp_arr_[ps_thread_index_]->SendMsg(m);
    }
  }

  if (need_exit) {
    return -1;
  }
  return 0;
}

INSTANTIATE_CLASS(ConvClient);

}  // end namespace caffe


