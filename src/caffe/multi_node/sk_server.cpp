
#include "caffe/multi_node/sk_server.hpp"

namespace caffe {

//TODO: remove the default ID
SkServer::SkServer()
  : SkSock(ZMQ_ROUTER, REQ_SERVER_ID)
{

}


SkServer::~SkServer()
{

}


int SkServer::Connect(string addr)
{
  LOG(ERROR) << "Cannot connect to " << addr << " in router";
  exit(-1);
}


shared_ptr<Msg> SkServer::RecvMsg(bool blocked)
{
  int src = 0;
  int r = 0;

  shared_ptr<Msg> m;

  zmq_msg_t zmsg;
  zmq_msg_init(&zmsg);
  
  if ( blocked ) {
    r = zmq_msg_recv (&zmsg, sock_, 0);
  } else {
    r = zmq_msg_recv (&zmsg, sock_, ZMQ_DONTWAIT);
    
    //return a null message
    if (r < 0) {
      m.reset();
      return m; 
    }
  }

  //deal with src, it is automatically added by zmq
  int size = zmq_msg_size (&zmsg);
  CHECK( size == sizeof(int) ) << "Please init zmq socket client id to a integer";
  memcpy( &src, zmq_msg_data(&zmsg), sizeof(int) );

  zmq_msg_close(&zmsg);
  
  //Make sure we have payloads
  CHECK(zmq_msg_more(&zmsg));
  
  return SkSock::RecvMsg(blocked);
}


int SkServer::SendMsg(shared_ptr<Msg> msg)
{
  //attaching dst
  zmq_msg_t header_dst;
  zmq_msg_init_size (&header_dst, sizeof(int));
  int dst = msg->dst();
  memcpy (zmq_msg_data (&header_dst), &dst, sizeof(int));
  zmq_msg_send (&header_dst, sock_, ZMQ_SNDMORE);

  return SkSock::SendMsg(msg);
}

} //end caffe

