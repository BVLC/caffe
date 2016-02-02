#ifndef CAFFE_INTERNODE_COMMUNICATION_H_
#define CAFFE_INTERNODE_COMMUNICATION_H_

#include <boost/function.hpp>
#include "caffe/blob.hpp"

namespace caffe {
namespace internode {

typedef size_t RemoteId;

class Waypoint {
 public:
  struct Handler {
    virtual void received(char* buffer, size_t size, RemoteId) = 0;
  };

  virtual void send(const char* buffer, size_t size) = 0;

  virtual void register_receive_handler(Handler* handler) = 0;
  virtual void close() = 0;
  virtual void shutdown() = 0;
};

class MultiWaypoint : public Waypoint {
 public:
  struct Handler {
    virtual void accepted(RemoteId) = 0;
    virtual void disconnected(RemoteId) = 0;
  };

  virtual void register_peer_change_handler(Handler* handler) = 0;
  virtual void send_to(const char* buffer, size_t size, RemoteId id) = 0;
};

}  // namespace internode
}  // namespace caffe

#endif  // CAFFE_INTERNODE_COMMUNICATION_H_

