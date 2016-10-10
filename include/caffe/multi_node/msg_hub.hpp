
#ifndef MULTI_NODE_MSG_HUB_H_
#define MULTI_NODE_MSG_HUB_H_

#include <boost/unordered_map.hpp>

#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/multi_node/cpu_dispatcher.hpp"
#include "caffe/multi_node/msg.hpp"
#include "caffe/multi_node/node_env.hpp"
#include "caffe/multi_node/sk_server.hpp"
#include "caffe/multi_node/worker_thread.hpp"

#define SERVER_SK_STR "inproc://workers"

using boost::unordered_map;

namespace caffe {
/**
* Virtual base class of convolution, FC and parameter server nodes
* It launches worker threads and set up in-process sockets for communication
*/
template <typename Dtype>
class MsgHub {
 public:
  MsgHub(int nthreads, int nworkers) {
    nthreads_ = nthreads;
    nworkers_ = nworkers;

    CHECK_GT(nthreads_, 0);
    CHECK_GT(nworkers_, 0);

    threads_.resize(nthreads);

    for (int i = 0; i < nthreads; i++) {
      sockp_arr_.push_back(shared_ptr<SkSock>(new SkSock(ZMQ_PAIR)));
      prior_socks_.push_back(shared_ptr<SkSock>(new SkSock(ZMQ_PAIR)));
    }

    // no poll items in the begining
    poll_items_ = NULL;
    num_poll_items_ = 0;

    // get ip address from machine (use the first interface by default)
    node_ip_ = NodeEnv::Instance()->IP();
    node_id_ = NodeEnv::Instance()->ID();

    string pub_addr = NodeEnv::Instance()->pub_addr();
    string back_addr = NodeEnv::Instance()->router_addr();

    if (!back_addr.empty()) {
      sock_back_.reset(new SkServer());
      sock_back_->Bind(back_addr);

      LOG(INFO) << "Bind Router to " << back_addr;
    }

    if (!pub_addr.empty()) {
      sock_pub_.reset(new SkSock(ZMQ_PUB));
      sock_pub_->Bind(pub_addr);
      LOG(INFO) << "Bind PUB to: " << pub_addr;
    }

    param_thread_index_ = nthreads - 1;
    back_sock_index_ = nthreads;
    sub_sock_index_ = back_sock_index_ + 1;

    int sub_addr_size = NodeEnv::Instance()->sub_addrs().size();
    poll_offset_ = sub_sock_index_ + sub_addr_size;
  }

  virtual ~MsgHub() {
    // deleting poll items if any
    if (NULL != poll_items_) {
      delete [] poll_items_;
    }
  }

  /// monitor the incoming and outgoing messages
  int Poll();

  // initing the contex for each thread, e.g. create caffe solver etc.
  // and connect the nodes together.
  virtual int Init() = 0;

  // scheduling incoming & out-going message
  virtual int RouteMsg() = 0;

 protected:
  virtual int SetUpPoll();

  // start all the work threads
  int StartThreads();

  // statically dispatch CPU cores to work threads
  void DispatchCPU(vector<vector<int> > *pthread_arr) {
    dispatcher_.Dispatch(pthread_arr);
  }

  // enqueue a message to a worker thread
  void Enqueue(int thrd_id, shared_ptr<Msg> m) {
    // prioritize processing of backward messages
    if (m->type() == BACKWARD) {
      prior_socks_[thrd_id]->SendMsg(m);
    } else {
      sockp_arr_[thrd_id]->SendMsg(m);
    }
    threads_[thrd_id]->Enqueue();
  }

  void BindCores(const vector<int>& core_list);

  void InitRoute();

  inline shared_ptr<SkSock> param_sock() {
    return sockp_arr_[param_thread_index_];
  }

  shared_ptr<SkSock> ConnectNode(const string& addr, int dst_id);

 protected:
  // total number of threads
  int nthreads_;

  // number of threads used as workers
  int nworkers_;

  vector<shared_ptr<WorkerThread<Dtype> > > threads_;

  // pair sockets to communicate with the work threads
  vector<shared_ptr<SkSock> > sockp_arr_;

  // send packets to worker with priority
  vector<shared_ptr<SkSock> > prior_socks_;

  // for message polling
  zmq_pollitem_t *poll_items_;

  // number of poll items
  int num_poll_items_;

  string node_ip_;

  CPUDispatcher dispatcher_;

  // broadcast blobs to downstream nodes
  shared_ptr<SkSock> sock_pub_;

  /// a ROUTER socket to received the packets from downstream nodes
  shared_ptr<SkSock> sock_back_;

  /// back sock index in the poll table
  int back_sock_index_;

  /// use hash map as a routing table
  /// map from node id to the sock index
  unordered_map<int, shared_ptr<SkSock> > node_to_sock_;

  /// NodeID, fetched from ID server
  int node_id_;

  //
  int param_thread_index_;

  // receive broadcast message from upstream nodes
  vector<shared_ptr<SkSock> > vec_sub_sock_;

  //
  int sub_sock_index_;

  // the poll offset used by MsgHub
  int poll_offset_;

 private:
  MsgHub() {  }

DISABLE_COPY_AND_ASSIGN(MsgHub);
};

}  // end namespace caffe


#endif


