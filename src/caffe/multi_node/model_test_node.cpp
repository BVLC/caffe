

#include "caffe/multi_node/model_test_node.hpp"

namespace caffe {

template <typename Dtype>
int TestClient<Dtype>::Init() {
  CHECK_EQ(ps_ids_.size(), ps_addrs_.size());

  for (int i = 0; i < ps_ids_.size(); i++) {
    shared_ptr<SkSock> sk(new SkSock(ZMQ_DEALER));
    sk->SetId(node_id_);
    sk->Connect(ps_addrs_[i]);
    node_id_to_sock_[ps_ids_[i]] = sk;
    ps_socks_.push_back(sk);
  }

  for (int i = 0; i < fc_ids_.size(); i++) {
    shared_ptr<SkSock> sk(new SkSock(ZMQ_DEALER));
    sk->SetId(node_id_);
    sk->Connect(fc_addrs_[i]);
    node_id_to_sock_[fc_ids_[i]] = sk;
    fc_socks_.push_back(sk);
  }

  for (int i = 0; i < this->nworkers_; i++) {
    this->threads_[i].reset(new TestThread<Dtype>());
  }

  return this->StartThreads();
}

template <typename Dtype>
int TestClient<Dtype>::SetUpPoll() {
  this->num_poll_items_ = this->nthreads_ + ps_ids_.size() + fc_ids_.size();
  this->poll_items_ = new zmq_pollitem_t[this->num_poll_items_];

  int poll_idx = this->nthreads_;

  for (int i = 0; i < ps_socks_.size(); i++, poll_idx++) {
    this->poll_items_[poll_idx].socket = ps_socks_[i]->GetSock();
    this->poll_items_[poll_idx].events = ZMQ_POLLIN;
    this->poll_items_[poll_idx].fd = 0;
    this->poll_items_[poll_idx].revents = 0;
  }

  for (int i = 0; i < fc_socks_.size(); i++, poll_idx++) {
    this->poll_items_[poll_idx].socket = fc_socks_[i]->GetSock();
    this->poll_items_[poll_idx].events = ZMQ_POLLIN;
    this->poll_items_[poll_idx].fd = 0;
    this->poll_items_[poll_idx].revents = 0;
  }

  return MsgHub<Dtype>::SetUpPoll();
}

template <typename Dtype>
int TestClient<Dtype>::RouteMsg() {
  bool need_exit = false;

  for (int i = 0; i < this->nthreads_; i++) {
    if (this->poll_items_[i].revents & ZMQ_POLLIN) {
      shared_ptr<Msg> m = this->sockp_arr_[i]->RecvMsg(true);
      if (m->type() == EXIT_TRAIN) {
        need_exit = true;
      } else {
        unordered_map<int, shared_ptr<SkSock> >::iterator iter =
                                            node_id_to_sock_.find(m->dst());

        CHECK(iter != node_id_to_sock_.end())
                            << "Cannot find route to node id: "
                            << m->dst();
        iter->second->SendMsg(m);
      }
    }
  }

  int poll_idx = this->nthreads_;
  for (int i = 0; i < ps_socks_.size(); i++, poll_idx++) {
    if (this->poll_items_[poll_idx].revents & ZMQ_POLLIN) {
      shared_ptr<Msg> m = ps_socks_[i]->RecvMsg(true);
      this->Enqueue(0, m);
    }
  }

  for (int i = 0; i < fc_socks_.size(); i++, poll_idx++) {
    if (this->poll_items_[poll_idx].revents & ZMQ_POLLIN) {
      shared_ptr<Msg> m = fc_socks_[i]->RecvMsg(true);
      this->Enqueue(0, m);
    }
  }

  if (need_exit) {
    return -1;
  }

  return 0;
}

INSTANTIATE_CLASS(TestClient);

}  // end namespace caffe



