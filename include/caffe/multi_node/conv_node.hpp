

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

    gateway_num_ = fc_clients_.size();

    fc_fwd_addrs_ = NodeEnv::Instance()->forward_addrs();
    fc_fwd_ids_ = NodeEnv::Instance()->forward_ids();

    fwd_socks_.resize(fc_fwd_ids_.size());

    // init parameter server addresses
    ps_addrs_ = NodeEnv::Instance()->ps_addrs();
    ps_ids_ = NodeEnv::Instance()->ps_ids();
    ps_num_ = ps_addrs_.size();

    ps_clients_.resize(ps_num_);

    fc_sock_index_ = this->poll_offset_;
    ps_sock_index_ = fc_sock_index_ + gateway_num_;
  }

  virtual ~ConvClient() { }

 public:
  virtual int Init();

  virtual int RouteMsg();

 protected:
  virtual int SetUpPoll();

  // send out the message to FC layer gateways
  void SendOutMsg(shared_ptr<Msg> m);

  // connect the topologies in reduce tree
  void SetUpReduceTree();

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

  /// the location of fc socket in zmq polling table
  int fc_sock_index_;

  /// the location of parameter server socket in zmq polling table
  int ps_sock_index_;

  // the route socket to receive packets
  shared_ptr<SkSock> back_sock_;

DISABLE_COPY_AND_ASSIGN(ConvClient);
};

}  // end namespace caffe

#endif


