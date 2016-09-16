#ifndef CAFFE_INTERNODE_TREE_CLUSTER_HPP_
#define CAFFE_INTERNODE_TREE_CLUSTER_HPP_

#include <boost/function.hpp>
#include <string>
#include <vector>

namespace caffe {
namespace internode {

class Daemon;

typedef size_t RemoteId;

class TreeWaypoint {
 public:
  struct Handler {
    virtual void received_from_parent(
            char* buffer, size_t size) = 0;
    virtual void received_from_child(
            char* buffer, size_t size, RemoteId id) = 0;
  };

  static TreeWaypoint* get_instance();

  virtual boost::shared_ptr<Daemon> get_daemon() = 0;
  virtual void set_buffer_size(size_t max_packet_size) = 0;

  typedef boost::function<void(bool succesful) > SentCallback;
  virtual void async_send_to_parent(
          const char* buffer, size_t size, SentCallback) = 0;
  virtual void async_send_to_children(
          const char* buffer, size_t size, SentCallback) = 0;

  virtual void register_receive_handler(Handler* handler) = 0;

  virtual RemoteId id() const = 0;
  virtual std::vector<RemoteId> children() const = 0;
  virtual RemoteId parent() const = 0;
  virtual int total_nodes() const = 0;
  virtual void lets_die_together() const = 0;
  virtual bool is_finished() const = 0;
};
}  // namespace internode
}  // namespace caffe

#endif  // CAFFE_INTERNODE_TREE_CLUSTER_HPP_
