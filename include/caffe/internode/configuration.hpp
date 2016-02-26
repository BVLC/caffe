#ifndef CAFFE_INTERNODE_CONFIGURATION_H_
#define CAFFE_INTERNODE_CONFIGURATION_H_

#include <boost/shared_ptr.hpp>
#include <string>
#include "communication.hpp"

namespace caffe {
namespace internode {

class Daemon;

typedef boost::function<void()> TimerCallback;

boost::shared_ptr<Daemon> create_communication_daemon();
void poll_one(boost::shared_ptr<Daemon>);
void run_one(boost::shared_ptr<Daemon>);
void create_timer(
  boost::shared_ptr<Daemon>,
  uint64_t duration_microseconds,
  TimerCallback callback,
  bool repeat);

boost::shared_ptr<Waypoint> configure_client(
    boost::shared_ptr<Daemon> communication_daemon,
    std::string address,
    size_t max_buffer_size);
boost::shared_ptr<MultiWaypoint> configure_server(
    boost::shared_ptr<Daemon> communication_daemon,
    std::string address,
    size_t max_buffer_size);

bool is_remote_address(std::string str);

}  // namespace internode
}  // namespace caffe

#endif  // CAFFE_INTERNODE_CONFIGURATION_H_

