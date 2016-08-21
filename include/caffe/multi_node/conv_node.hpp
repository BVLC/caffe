

#ifndef MULTI_NODE_CONV_NODE_H_
#define MULTI_NODE_CONV_NODE_H_

#include <string>
#include <vector>

#include "caffe/multi_node/msg_hub.hpp"
#include "caffe/multi_node/param_helper.hpp"

namespace caffe {

/*
 * Nodes responsible for convolution
 * Several work threads are forked to do convolution
 * The main thread is used to route messages
 */
template <typename Dtype>
class ConvClient : public MsgHub<Dtype> {
 public:
  explicit ConvClient(int nthreads)
    : MsgHub<Dtype>(nthreads, nthreads - 1) {
    fc_gateway_addrs_ = NodeEnv::Instance()->gateway_addrs();
    fc_gateway_ids_ = NodeEnv::Instance()->gateway_ids();

    fc_clients_.resize(fc_gateway_addrs_.size());
    int client_id = NodeEnv::Instance()->ID();

    for (int i = 0; i < fc_clients_.size(); i++) {
      fc_clients_[i].reset(new SkSock(ZMQ_DEALER));
      fc_clients_[i]->SetId(client_id);
    }
    gateway_num_ = fc_clients_.size();

    fc_fwd_addrs_ = NodeEnv::Instance()->forward_addrs();
    fc_fwd_ids_ = NodeEnv::Instance()->forward_ids();

    fwd_socks_.resize(fc_fwd_ids_.size());
    for (int i = 0; i < fwd_socks_.size(); i++) {
      fwd_socks_[i].reset(new SkSock(ZMQ_DEALER));
      fwd_socks_[i]->SetId(client_id);
    }

    // init parameter server addresses
    ps_addrs_ = NodeEnv::Instance()->ps_addrs();
    ps_ids_ = NodeEnv::Instance()->ps_ids();
    ps_num_ = ps_addrs_.size();

    // Connect to parameter server
    for (int i = 0; i < ps_num_; i++) {
      shared_ptr<SkSock> ps_sock(new SkSock(ZMQ_DEALER));
      ps_sock->SetId(client_id);
      ps_clients_.push_back(ps_sock);
    }

    fc_sock_index_ = nthreads;
    ps_sock_index_ = nthreads + gateway_num_;
    ps_thread_index_ = nthreads - 1;
  }

  virtual ~ConvClient() { }

 public:
  virtual int Init();

  virtual int RouteMsg();

 protected:
  virtual int SetUpPoll();

  // send out the message to FC layer gateways
  void SendOutMsg(shared_ptr<Msg> m);

 protected:
  /// socket used to communicate Fully Connected layers
  vector<shared_ptr<SkSock> > fc_clients_;

  /// zmq addr of Fully Connected Layers gateway
  vector<string> fc_gateway_addrs_;

  vector<int> fc_gateway_ids_;

  vector<int> fc_fwd_ids_;

  vector<string> fc_fwd_addrs_;

  vector<shared_ptr<SkSock> > fwd_socks_;

  // number of gateways we have
  int gateway_num_;

  /// zmq addr of Parameter Server
  vector<string> ps_addrs_;

  // the node id of parameter server
  vector<int> ps_ids_;

  // number of parameter servers
  int ps_num_;

  /// socket used to communicate parameter server
  vector<shared_ptr<SkSock> > ps_clients_;

  // map node id to sock
  unordered_map<int, shared_ptr<SkSock> > node_to_sock_;

  /// the location of fc socket in zmq polling table
  int fc_sock_index_;
  /// the location of parameter server socket in zmq polling table
  int ps_sock_index_;

  /// A thread is dedicated to communicate parameter server
  /// this index stores its location in the polling table
  int ps_thread_index_;

DISABLE_COPY_AND_ASSIGN(ConvClient);
};

}  // end namespace caffe

#endif


