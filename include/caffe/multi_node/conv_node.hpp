

#ifndef MULTI_NODE_CONV_NODE_H_
#define MULTI_NODE_CONV_NODE_H_

#include "caffe/multi_node/msg_hub.hpp"

namespace caffe {

/*
 * Nodes responsible for convolution
 * Several work threads are forked to do convolution
 * The main thread is used to route messages
 */
template <typename Dtype>
class ConvClient : public MsgHub<Dtype>
{

public:
  ConvClient(int nthreads, const string& fc_gateway_addr, const string& ps_addr)
    : MsgHub<Dtype>(nthreads, nthreads - 1)
  {
    fc_client_.reset(new SkSock(ZMQ_DEALER));
    
    int client_id = NodeEnv::Instance()->ID();
    fc_client_->SetId(client_id);

    ps_client_.reset(new SkSock(ZMQ_DEALER));
    ps_client_->SetId(client_id);

    fc_gateway_addr_ = fc_gateway_addr;
    ps_addr_ = ps_addr;

    fc_sock_index_ = nthreads;
    ps_sock_index_ = nthreads + 1;
    ps_thread_index_ = nthreads - 1;
  }


  virtual ~ConvClient() { }

public:
  virtual int Init();

  virtual int RouteMsg();

protected:
  virtual int SetUpPoll();
  

protected:
  /// socket used to communicate Fully Connected layers
  shared_ptr<SkSock> fc_client_;
  /// socket used to communicate parameter server
  shared_ptr<SkSock> ps_client_;
  /// zmq addr of Fully Connected Layers gateway
  string fc_gateway_addr_;
  /// zmq addr of Parameter Server
  string ps_addr_;
  
  /// the location of fc socket in zmq polling table
  int fc_sock_index_;
  /// the location of parameter server socket in zmq polling table
  int ps_sock_index_;
  /// A thread is dedicated to communicate parameter server
  /// this index stores its location in the polling table
  int ps_thread_index_;

DISABLE_COPY_AND_ASSIGN(ConvClient);
};

} //end caffe

#endif


