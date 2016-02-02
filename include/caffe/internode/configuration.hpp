#ifndef CAFFE_INTERNODE_CONFIGURATION_H_
#define CAFFE_INTERNODE_CONFIGURATION_H_

#include <boost/shared_ptr.hpp>
#include <string>
#include "communication.hpp"

namespace caffe {
namespace internode {

class Daemon;
boost::shared_ptr<Daemon> create_communication_daemon();
void run(boost::shared_ptr<Daemon>);
void run_one(boost::shared_ptr<Daemon>);
void poll_one(boost::shared_ptr<Daemon>);
void poll(boost::shared_ptr<Daemon>);

boost::shared_ptr<Waypoint> configure_client(
    boost::shared_ptr<Daemon> communication_daemon,
    std::string address);

boost::shared_ptr<MultiWaypoint> configure_server(
    boost::shared_ptr<Daemon> communication_daemon,
    std::string address);

bool is_remote_address(std::string str);

}  // namespace internode
}  // namespace caffe

#endif  // CAFFE_INTERNODE_CONFIGURATION_H_
