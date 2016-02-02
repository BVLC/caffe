#ifndef CAFFE_INTERNODE_COMMUNICATION_H_
#define CAFFE_INTERNODE_COMMUNICATION_H_

#include <boost/function.hpp>
#include <string>
#include "caffe/blob.hpp"

namespace caffe {
namespace internode {

typedef size_t RemoteId;

class Waypoint {
 public:
  struct Handler {
    virtual void received(char* buffer, size_t size, Waypoint*) = 0;
  };

  typedef boost::function<void(bool succesful)> SentCallback;
  virtual void async_send(const char* buffer, size_t size, SentCallback) = 0;

  virtual void register_receive_handler(Handler* handler) = 0;

  virtual RemoteId id() const = 0;
  virtual string address() const = 0;
  virtual bool guaranteed_comm() const = 0;
  virtual size_t max_packet_size() const = 0;
};

class MultiWaypoint : public Waypoint {
 public:
  struct Handler {
    virtual void accepted(shared_ptr<Waypoint>) = 0;
    virtual void disconnected(RemoteId) = 0;
  };

  virtual void register_peer_change_handler(Handler* handler) = 0;
  virtual size_t max_packet_size() const = 0;
};

}  // namespace internode
}  // namespace caffe

#endif  // CAFFE_INTERNODE_COMMUNICATION_H_

