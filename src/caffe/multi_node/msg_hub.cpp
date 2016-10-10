
#include <string>
#include <vector>

#include "caffe/multi_node/msg_hub.hpp"

namespace caffe {

template <typename Dtype>
int MsgHub<Dtype>::StartThreads() {
  for (int i = 0; i < nthreads_; i++) {
    string sk_addr(SERVER_SK_STR);
    sk_addr += "_";
    sk_addr += boost::lexical_cast<string>(i);

    string prior_addr = sk_addr + "_prior";

    sockp_arr_[i]->Bind(sk_addr);
    prior_socks_[i]->Bind(prior_addr);

    threads_[i]->SetWorkerId(i);
    threads_[i]->SetWorkerNum(nworkers_);
    threads_[i]->SetClientAddr(sk_addr);
    threads_[i]->SetPriorAddr(prior_addr);

    threads_[i]->StartInternalThread();
  }

  return 0;
}

template <typename Dtype>
void MsgHub<Dtype>::InitRoute() {
  // Connect Upstream PUB sockets
  const vector<string>& sub_addrs = NodeEnv::Instance()->sub_addrs();

  for (int i = 0; i < sub_addrs.size(); i++) {
    // connect PUB node
    shared_ptr<SkSock> sub(new SkSock(ZMQ_SUB));
    sub->Connect(sub_addrs[i]);
    zmq_setsockopt(sub->GetSock(), ZMQ_SUBSCRIBE, "", 0);

    vec_sub_sock_.push_back(sub);
  }

  const vector<string>& prev_addrs = NodeEnv::Instance()->prev_router_addrs();
  const vector<int>& prev_ids = NodeEnv::Instance()->prev_node_ids();

  for (int i = 0; i < prev_addrs.size(); i++) {
    ConnectNode(prev_addrs[i], prev_ids[i]);
  }

  return;
}

template <typename Dtype>
shared_ptr<SkSock> MsgHub<Dtype>::ConnectNode(const string& addr,
                                              int dst_id) {
  unordered_map<int, shared_ptr<SkSock> >::iterator iter =
                                                node_to_sock_.find(dst_id);
  if (iter != node_to_sock_.end()) {
    return iter->second;
  }

  shared_ptr<SkSock> dealer(new SkSock(ZMQ_DEALER));
  dealer->SetId(node_id_);
  dealer->Connect(addr);
  node_to_sock_[dst_id] = dealer;

  return dealer;
}

template <typename Dtype>
int MsgHub<Dtype>::SetUpPoll() {
  if (poll_items_ == NULL) {
    num_poll_items_ = poll_offset_ + 1;
    poll_items_ = new zmq_pollitem_t[this->num_poll_items_];
    memset(poll_items_, 0, sizeof(zmq_pollitem_t) * this->num_poll_items_); // NOLINT
  }
  // initialize polling items for the work threads
  for (int i = 0; i < nthreads_; i++) {
    poll_items_[i].socket = sockp_arr_[i]->GetSock();
    poll_items_[i].events = ZMQ_POLLIN;
    poll_items_[i].fd = 0;
    poll_items_[i].revents = 0;
  }

  if (sock_back_ != NULL) {
    poll_items_[back_sock_index_].socket = sock_back_->GetSock();
    poll_items_[back_sock_index_].events = ZMQ_POLLIN;
    poll_items_[back_sock_index_].fd = 0;
    poll_items_[back_sock_index_].revents = 0;
  }

  // adding the subscribers to the polling items
  int poll_index = sub_sock_index_;
  for (int i = 0; i < vec_sub_sock_.size(); i++, poll_index++) {
    this->poll_items_[poll_index].socket = vec_sub_sock_[i]->GetSock();
    this->poll_items_[poll_index].events = ZMQ_POLLIN;
    this->poll_items_[poll_index].fd = 0;
    this->poll_items_[poll_index].revents = 0;
  }

  return 0;
}

// dispatch thread via poll
template <typename Dtype>
int MsgHub<Dtype>::Poll() {
  #ifdef USE_MKL
  // only use 1 mkl thread to avoid contention
  mkl_set_num_threads_local(1);
  #endif

  SetUpPoll();

  while (true) {
    // blocked poll
    zmq_poll(poll_items_, num_poll_items_, -1);

    if (RouteMsg() < 0) {
      break;
    }
  }

  return 0;
}

template <typename Dtype>
void MsgHub<Dtype>::BindCores(const vector<int>& core_list) {
  cpu_set_t new_mask;
  CPU_ZERO(&new_mask);
  for (int i = 0; i < core_list.size(); i++) {
    CPU_SET(core_list[i], &new_mask);
  }

  if (sched_setaffinity(0, sizeof(new_mask), &new_mask) == -1) {
    LOG(ERROR) << "cannot bind to cores";
  }
}


INSTANTIATE_CLASS(MsgHub);

}  // end namespace caffe

