
#ifndef MULTI_NODE_FC_NODE_H_
#define MULTI_NODE_FC_NODE_H_

#include "caffe/multi_node/msg_hub.hpp"
#include "caffe/multi_node/fc_thread.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
class FcNode : public MsgHub<Dtype>
{

public:
  FcNode (int nthreads, int nworkers) 
      : MsgHub<Dtype>(nthreads, nworkers)
  {
    node_id_ = NodeEnv::Instance()->ID();

    string pub_addr = NodeEnv::Instance()->PubAddr();
    string back_addr = NodeEnv::Instance()->RouterAddr();

    sock_pub_.reset(new SkSock(ZMQ_PUB));
    sock_pub_->Bind(pub_addr);

    LOG(INFO) << "Bind PUB to: " << pub_addr;

    sock_back_.reset(new SkServer());
    sock_back_->Bind(back_addr);
    
    LOG(INFO) << "Bind Router to " << back_addr;

    num_next_hops_ = 0;
    param_thread_index_ = nthreads - 1;
    back_sock_index_ = nthreads;
  }


public:
  virtual int Init();

  virtual int RouteMsg();


protected:
  virtual int SetUpPoll();
  
  //send out the msg processed by threads
  virtual int SendOutMsg(shared_ptr<Msg> m);
  
  //Set up the connections and init the routing table 
  int InitRoute();
  
protected:
  shared_ptr<SkSock> sock_pub_;

  //a REP socket to received the packets from downstream nodes
  shared_ptr<SkSock> sock_back_;
  //back sock index in the poll table
  int back_sock_index_;
  int num_next_hops_;
  
  //use hash map as a routing table
  //map from node id to the sock index
  unordered_map<int, shared_ptr<SkSock> > node_to_sock_;
  
  //the dealer sock
  vector<shared_ptr<SkSock> > vec_dealer_;
  
  //NodeID, fetched from ID server
  int node_id_;

  //
  int param_thread_index_;
};


//Interface to the conv. clients
template <typename Dtype>
class FcGateway : public FcNode<Dtype>
{
public:
  FcGateway(int nthreads, string server_addr)
      : FcNode<Dtype>(nthreads, nthreads - 1)
  {
    sock_server_.reset(new SkServer());
    sock_server_->Bind(server_addr);

    msg_id_ = 0;
    server_sock_index_ = nthreads + 1;
  }

public:
  virtual int Init();

  virtual int RouteMsg();

  virtual int SendOutMsg(shared_ptr<Msg> m);

protected:
  virtual int SetUpPoll();

protected:
  shared_ptr<SkServer> sock_server_;
  
  //position of server sock
  int server_sock_index_;

  //increasing message id by 1 at a time
  int64_t msg_id_;
  
  //connect to the nodes in bottom layer
  vector<shared_ptr<SkSock> > bottom_socks_;
};

template <typename Dtype>
class FcClient : public FcNode<Dtype>
{
public:
  FcClient(int nthreads)
      : FcNode<Dtype>(nthreads, nthreads - 1)
  {
    sub_sock_index_ = nthreads + 1;
  }

public:
  virtual int Init();
  virtual int RouteMsg();

protected:
  virtual int SetUpPoll();

protected:
  //receive broadcast message from upstream nodes
  vector<shared_ptr<SkSock> > vec_sub_sock_;
  
  //
  int sub_sock_index_;
};

} //end caffe

#endif


