
#ifndef MULTI_NODE_FC_NODE_H_
#define MULTI_NODE_FC_NODE_H_

#include "caffe/multi_node/msg_hub.hpp"
#include "caffe/multi_node/fc_thread.hpp"
#include "caffe/sgd_solvers.hpp"
#include "boost/lexical_cast.hpp"

namespace caffe {

template <typename Dtype>
class FcNode : public MsgHub<Dtype>
{

public:
  FcNode (int nthreads, int nworkers) 
      : MsgHub<Dtype>(nthreads, nworkers)
  {
    node_id_ = NodeEnv::Instance()->ID();

    string pub_addr = NodeEnv::Instance()->pub_addr();
    string back_addr = NodeEnv::Instance()->router_addr();

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
  
  /// send out the msg processed by threads
  virtual int SendOutMsg(shared_ptr<Msg> m);
  
  /// Set up the connections and init the routing table 
  int InitRoute();

  inline int num_inputs() { return input_blob_name_map_.size(); }
  
  bool is_input_blob(const string& blob_name) {
    unordered_map<string, bool>::iterator iter = input_blob_name_map_.find(blob_name);
    if (iter == input_blob_name_map_.end()) {
      return false;
    } else {
      return iter->second;
    }
  }

  void PrepareInputData(shared_ptr<Msg> m);

protected:
  // broadcast blobs to downstream nodes
  shared_ptr<SkSock> sock_pub_;

  /// a ROUTER socket to received the packets from downstream nodes
  shared_ptr<SkSock> sock_back_;
  
  /// back sock index in the poll table
  int back_sock_index_;
  int num_next_hops_;
  
  /// use hash map as a routing table
  /// map from node id to the sock index
  unordered_map<int, shared_ptr<SkSock> > node_to_sock_;
  
  /// the dealer socks used to connect upstream nodes
  vector<shared_ptr<SkSock> > vec_dealer_;
  
  /// NodeID, fetched from ID server
  int node_id_;
  
  // use hash map to store message
  unordered_map<int64_t, shared_ptr<Msg> > id_to_msg_;
  
  unordered_map<string, bool> input_blob_name_map_;

  //
  int param_thread_index_;
};


//Interface to the conv. clients
template <typename Dtype>
class FcGateway : public FcNode<Dtype>
{
public:
  FcGateway(int nthreads)
      : FcNode<Dtype>(nthreads, nthreads - 1)
  {
    gateway_addr_ = "tcp://*:";
    gateway_addr_ += boost::lexical_cast<string>(GATEWAY_PORT);
    sock_server_.reset(new SkServer());
    sock_server_->Bind(gateway_addr_);

    msg_id_ = 0;
    server_sock_index_ = nthreads + 1;
  }

public:
  virtual int Init();

  virtual int RouteMsg();

  virtual int SendOutMsg(shared_ptr<Msg> m);

protected:
  virtual int SetUpPoll();

  void DemuxMsg(shared_ptr<Msg> m);

protected:
  string gateway_addr_;

  shared_ptr<SkServer> sock_server_;
  
  //position of server sock
  int server_sock_index_;

  //increasing message id by 1 at a time
  int64_t msg_id_;
  
  // sockets to forward blobs
  vector<shared_ptr<SkSock> > fwrd_socks_;
  
  // map a forward blob to a list of ids
  unordered_map<string, shared_ptr<vector<int> > > fwrd_blob_name_to_ids_;

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


