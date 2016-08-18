

#ifndef MULTI_NODE_SK_SERVER_H
#define MULTI_NODE_SK_SERVER_H

#include <string>

#include "caffe/multi_node/msg.hpp"
#include "caffe/multi_node/sk_sock.hpp"

namespace caffe {
/*
* SkServer wrapps ZMQ router socket
* ZMQ attaches an ID in a message sent by a dealer socket
* to router socket, we intentionally remove the header
* and hide the details to high lever classes
*/
class SkServer : public SkSock {
 public:
  SkServer();

  virtual ~SkServer();

  virtual int Connect(string addr);

  /// removes the ID message added by ZMQ layer
  virtual shared_ptr<Msg> RecvMsg(bool blocked);

  /// attaches an ID message for ZMQ
  virtual int SendMsg(shared_ptr<Msg> msg);
};

}  // end namespace caffe

#endif


