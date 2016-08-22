
#ifndef MULTI_NODE_FC_NODE_H_
#define MULTI_NODE_FC_NODE_H_

#include <boost/lexical_cast.hpp>

#include <string>
#include <vector>

#include "caffe/multi_node/fc_thread.hpp"
#include "caffe/multi_node/msg_hub.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
class FcNode : public MsgHub<Dtype> {
 public:
  FcNode(int nthreads, int nworkers, int omp_param_threads = 0)
      : MsgHub<Dtype>(nthreads, nworkers),
      work_loads_(nworkers, 0) {
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

    omp_param_threads_ = omp_param_threads;
  }

 public:
  virtual int Init();

  virtual int RouteMsg();

 protected:
  virtual int SetUpPoll();

  /// send out the msg processed by threads
  virtual int SendOutMsg(shared_ptr<Msg> m);

  virtual void ProcessFwdMsg(shared_ptr<Msg> m);

  /// Set up the connections and init the routing table
  int InitRoute();

  inline int num_inputs() { return input_blob_name_map_.size(); }

  bool is_input_blob(const string& blob_name) {
    unordered_map<string, bool>::iterator iter =
                                 input_blob_name_map_.find(blob_name);
    if (iter == input_blob_name_map_.end()) {
      return false;
    } else {
      return iter->second;
    }
  }

  void PrepareInputData(shared_ptr<Msg> m);

  // schedule the message to worker threads
  int ScheduleMsg(shared_ptr<Msg> m);

 protected:
  // broadcast blobs to downstream nodes
  shared_ptr<SkSock> sock_pub_;

  /// a ROUTER socket to received the packets from downstream nodes
  shared_ptr<SkSock> sock_back_;

  /// back sock index in the poll table
  int back_sock_index_;
  int num_next_hops_;

  vector<int> prev_ids_;
  int num_prev_hops_;

  /// use hash map as a routing table
  /// map from node id to the sock index
  unordered_map<int, shared_ptr<SkSock> > node_to_sock_;

  /// a thread handles a client
  unordered_map<int, int> src_to_thread_;

  /// number of conv. clients a thread handles
  vector<int> work_loads_;

  /// the dealer socks used to connect upstream nodes
  vector<shared_ptr<SkSock> > vec_dealer_;

  /// NodeID, fetched from ID server
  int node_id_;

  typedef unordered_map<int64_t, shared_ptr<vector<shared_ptr<Msg> > > >
                                                                  MsgMap;

  // map msg id to a vector of partial message
  MsgMap msg_id_to_buf_;

  // use hash map to store message
  unordered_map<int64_t, shared_ptr<Msg> > id_to_msg_;

  unordered_map<string, bool> input_blob_name_map_;

  //
  int param_thread_index_;

  // number of omp threads for param thread
  int omp_param_threads_;
};


// Interface to the conv. clients
template <typename Dtype>
class FcGateway : public FcNode<Dtype> {
 public:
  FcGateway(int nthreads, int omp_param_threads = 0)
      : FcNode<Dtype>(nthreads, nthreads - 1, omp_param_threads) { }

 public:
  virtual int SendOutMsg(shared_ptr<Msg> m);

  virtual int RouteMsg() {
    return FcNode<Dtype>::RouteMsg();
  }

 protected:
  virtual int SetUpPoll();
};

template <typename Dtype>
class FcClient : public FcNode<Dtype> {
 public:
  FcClient(int nthreads, int omp_param_threads = 0)
      : FcNode<Dtype>(nthreads, nthreads - 1, omp_param_threads) {
    sub_sock_index_ = nthreads + 1;
  }

 public:
  virtual int Init();
  virtual int RouteMsg();

 protected:
  virtual int SetUpPoll();

 protected:
  // receive broadcast message from upstream nodes
  vector<shared_ptr<SkSock> > vec_sub_sock_;

  //
  int sub_sock_index_;
};

}  // end namespace caffe

#endif


