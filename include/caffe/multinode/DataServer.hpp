#ifndef CAFFE_MULTINODE_DATA_SERVER_HPP_
#define CAFFE_MULTINODE_DATA_SERVER_HPP_

#include <set>
#include <string>
#include <vector>
#include "caffe/internode/configuration.hpp"
#include "caffe/solver.hpp"

namespace caffe {

template <typename Dtype>
class DataServer : public internode::Waypoint::Handler {
  shared_ptr<internode::Daemon> daemon;
  shared_ptr<Solver<Dtype> > solver;
  shared_ptr<internode::MultiWaypoint> waypoint;

 public:
  DataServer(shared_ptr<Solver<Dtype> >, string bind_address);
  void run();

  virtual void received(char* buffer, size_t size, internode::RemoteId);
};

}  // namespace caffe

#endif  // CAFFE_MULTINODE_DATA_SERVER_HPP_

