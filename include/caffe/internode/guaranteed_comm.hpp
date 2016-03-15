#ifndef CAFFE_INTERNODE_GUARANTEED_COMM_H_
#define CAFFE_INTERNODE_GUARANTEED_COMM_H_

#include <boost/shared_ptr.hpp>
#include "configuration.hpp"

namespace caffe {
namespace internode {

boost::shared_ptr<Waypoint> configure_guaranteed_client(
  boost::shared_ptr<Daemon> external_daemon,
  boost::shared_ptr<Daemon> internal_daemon,
  boost::shared_ptr<Waypoint> client);
boost::shared_ptr<MultiWaypoint> configure_guaranteed_server(
  boost::shared_ptr<Daemon> external_daemon,
  boost::shared_ptr<Daemon> internal_daemon,
  boost::shared_ptr<MultiWaypoint> server);

}  // namespace internode
}  // namespace caffe

#endif  // CAFFE_INTERNODE_GUARANTEED_COMM_H_

