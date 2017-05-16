

#ifndef MULTI_NODE_SK_SOCK_H
#define MULTI_NODE_SK_SOCK_H

#include "caffe/multi_node/msg.hpp"
#include <boost/thread.hpp>

namespace caffe {

class SkSock {

public:
  SkSock() {
    id_ = INVALID_ID;
  }

  SkSock(int type, int id) {
    InitZmq(); 
    sock_ = zmq_socket (zmq_ctx_, type);
    id_ = id;
    sk_type_ = type;
  }
  
  SkSock(int type) {
    InitZmq(); 
    sk_type_ = type;
    sock_ = zmq_socket (zmq_ctx_, type);
  }

  virtual ~SkSock() {
    if (NULL != sock_) {
        zmq_close(sock_);
    }
    
    boost::mutex::scoped_lock lock(zmq_init_mutex_);
    LOG(INFO) << "deinitialize zmq ctx cnt: " << inited_cnt_;
    
    inited_cnt_--;
    if (0 == inited_cnt_) {
        LOG(INFO) << "destroying zmq context";
        zmq_ctx_destroy(zmq_ctx_);
    }

  }
  
  virtual int Bind(const string& addr);

  virtual int Connect(const string& addr);
  
  virtual shared_ptr<Msg> RecvMsg(bool blocked);

  virtual int SendMsg(shared_ptr<Msg> msg);

  int GetId() { return id_; }

  int SetId(int id) {
    id_ = id;
    zmq_setsockopt (sock_, ZMQ_IDENTITY, &id, sizeof(id));

    return 0;
  }
  
  /// return the native zmq socket
  void *GetSock() {
    return sock_;
  }

protected:
  int SendHeader(shared_ptr<Msg> msg);

protected:
  // the native zmq socket
  void *sock_;
  // the zmq address of this socket
  string addr_;

  void InitZmq(void);
  static void *zmq_ctx_;

protected:
  // an unique id allocated by an global id server
  int id_;
  // zmq socket type
  int sk_type_;

private:
  // a zmq context should be inited only once
  static boost::mutex zmq_init_mutex_;
  static int inited_cnt_;
};

} //end caffe

#endif




