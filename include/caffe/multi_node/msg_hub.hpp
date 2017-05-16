
#ifndef MULTI_NODE_MSG_HUB_H_
#define MULTI_NODE_MSG_HUB_H_

#include "caffe/multi_node/msg.hpp"
#include "caffe/multi_node/sk_server.hpp"
#include "caffe/multi_node/worker_thread.hpp"
#include "caffe/caffe.hpp"

#include "caffe/multi_node/node_env.hpp"

#define SERVER_SK_STR "inproc://workers"

#include "boost/unordered_map.hpp"

using boost::unordered_map;

namespace caffe {
/**
* Virtual base class of convolution, FC and parameter server nodes
* It launches worker threads and set up in-process sockets for communication
*/
template <typename Dtype>
class MsgHub
{

public:
  MsgHub(int nthreads, int nworkers) {
    nthreads_ = nthreads;
    nworkers_ = nworkers;
    
    CHECK_GT(nthreads_, 0);
    CHECK_GT(nworkers_, 0);

    threads_.resize(nthreads);

    for (int i = 0; i < nthreads; i++) {
      sockp_arr_.push_back(shared_ptr<SkSock>(new SkSock(ZMQ_PAIR)));
    }

    //no poll items in the begining
    poll_items_ = NULL;
    num_poll_items_ = 0;
    
    //get ip address from machine (use the first interface by default)
    node_ip_ = NodeEnv::Instance()->IP();
  }
  
  virtual ~MsgHub() {
    //deleting poll items if any
    if ( NULL != poll_items_ ) {
      delete [] poll_items_;
    }
  }
  
  /// monitor the incoming and outgoing messages
  int Poll();
  
  //initing the contex for each thread, e.g. create caffe solver etc.
  //and connect the nodes together.
  virtual int Init() = 0;

  //scheduling incoming & out-going message
  virtual int RouteMsg() = 0;
  
  //use the internal threads to process message
  int ProcessMsg(shared_ptr<Msg> m);

protected:
  virtual int SetUpPoll();
  int StartThreads();

protected:
  //total number of threads
  int nthreads_;
  //number of threads used as workers
  int nworkers_;
  vector<shared_ptr<WorkerThread<Dtype> > > threads_;

  //pair sockets to communicate with the work threads
  vector<shared_ptr<SkSock> > sockp_arr_;

  //for message polling
  zmq_pollitem_t *poll_items_;
  int num_poll_items_;

  string node_ip_;
  
  //use hash map to store message
  unordered_map<int64_t, shared_ptr<Msg> > id_to_msg_;

private:
  MsgHub() {  }

DISABLE_COPY_AND_ASSIGN(MsgHub);
};

} // end caffe


#endif


