#ifndef CAFFE_INTERNODE_TCP_CONFIGURATION_H_
#define CAFFE_INTERNODE_TCP_CONFIGURATION_H_

#include <boost/asio.hpp>
#include <boost/shared_ptr.hpp>
#include <string>
#include "configuration.hpp"

namespace caffe {
namespace internode {

boost::shared_ptr<Waypoint> configure_tcp_client(
    boost::shared_ptr<Daemon> communication_daemon,
    std::string ip,
    std::string port);
boost::shared_ptr<MultiWaypoint> configure_tcp_server(
    boost::shared_ptr<Daemon> communication_daemon,
    std::string port);

}  // namespace internode
}  // namespace caffe

#endif  // CAFFE_INTERNODE_TCP_CONFIGURATION_H_

