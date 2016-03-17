#ifndef CAFFE_INTERNODE_UDP_CONFIGURATION_H_
#define CAFFE_INTERNODE_UDP_CONFIGURATION_H_

#include <boost/shared_ptr.hpp>
#include <string>
#include "communication.hpp"

namespace caffe {
namespace internode {

boost::shared_ptr<Waypoint> configure_udp_client(
    boost::shared_ptr<Daemon> communication_daemon,
    std::string ip,
    std::string port,
    std::string group_ip,
    std::string group_port,
    size_t max_buffer_size);
boost::shared_ptr<MultiWaypoint> configure_udp_server(
    boost::shared_ptr<Daemon> communication_daemon,
    std::string ip,
    std::string port,
    std::string group_ip,
    std::string group_port,
    size_t max_buffer_size);

}  // namespace internode
}  // namespace caffe

#endif  // CAFFE_INTERNODE_UDP_CONFIGURATION_H_

