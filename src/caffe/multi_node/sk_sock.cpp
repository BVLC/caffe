

#include "caffe/multi_node/sk_sock.hpp"

namespace caffe {

void *SkSock::zmq_ctx_ = NULL;
boost::mutex SkSock::zmq_init_mutex_;
int SkSock::inited_cnt_ = 0;

void SkSock::InitZmq(void)
{
  boost::mutex::scoped_lock lock(zmq_init_mutex_);

  if (0 == inited_cnt_) {
      zmq_ctx_ = zmq_ctx_new();
  }

  inited_cnt_++;
}


inline int SkSock::SendHeader(shared_ptr<Msg> msg)
{
  //ataching meta data, including type, src, dst etc.
  string header_str;
  msg->SerializeHeader(&header_str);

  zmq_msg_t header_msg;
  zmq_msg_init_size (&header_msg, header_str.length());

  memcpy (zmq_msg_data (&header_msg), header_str.data(), header_str.length());
  zmq_msg_send (&header_msg, sock_, ZMQ_SNDMORE);
  
  return 0;
}


int SkSock::SendMsg(shared_ptr<Msg> msg)
{
  if (sk_type_ == ZMQ_DEALER) {
    CHECK_GT(msg->src(), 0) << "Message source must be set for dealer socket";
  }
  
  if (msg->ZmsgCnt() <= 0) {
    LOG(WARNING) << "Cannot send message with length " << msg->ZmsgCnt();

    return -1;
  }
  
  SendHeader(msg);

  for (int i = 0; i < msg->ZmsgCnt() - 1; i++) {
    zmq_msg_send(msg->GetZmsg(i), sock_, ZMQ_SNDMORE);
  }

  zmq_msg_send(msg->GetZmsg(msg->ZmsgCnt() - 1), sock_, 0);

  return 0;
}


shared_ptr<Msg> SkSock::RecvMsg(bool blocked)
{
  shared_ptr<Msg> m(new Msg());

  int r = 0;

  zmq_msg_t header_msg;
  zmq_msg_init(&header_msg);
  if ( blocked ) {
    r = zmq_msg_recv (&header_msg, sock_, 0);
  } else {
    r = zmq_msg_recv (&header_msg, sock_, ZMQ_DONTWAIT);

    if (r < 0) {
        m.reset();
        return m;
    }
  }
  
  string header_str( string( (char *) zmq_msg_data(&header_msg), zmq_msg_size (&header_msg) ) );

  m->ParseHeader(header_str);
  zmq_msg_close(&header_msg);
  
  CHECK(zmq_msg_more (&header_msg));

  for (int i = 0; ; i++) {
    zmq_msg_t *zmsg = new zmq_msg_t;
    zmq_msg_init(zmsg);
    
    if ( blocked ) {
      r = zmq_msg_recv (zmsg, sock_, 0);
    } else {
      r = zmq_msg_recv (zmsg, sock_, ZMQ_DONTWAIT);    
    }

    CHECK( r >= 0 );

    m->AppendZmsg(zmsg);
        
    if (! zmq_msg_more(zmsg)) {
      break;
    }
  }

  return m;
}

int SkSock::Bind(const string& addr)
{
  CHECK(sock_ != NULL) << "cannot bind null sock";

  int rc = 0;
  rc = zmq_bind(sock_, addr.c_str());
  CHECK( 0 == rc) << "failed to bind zmq sock to " << addr;

  addr_ = addr;

  return 0;
}


int SkSock::Connect(const string& addr)
{
  CHECK (sock_ != NULL) << "cannot connect to NULL sock";

  zmq_connect(sock_, addr.c_str());
  addr_ = addr;

  return 0;
}

}  //end caffe

