#ifndef CAFFE_SYNCHRONOUSPARAMSERVER_HPP_
#define CAFFE_SYNCHRONOUSPARAMSERVER_HPP_

#include <string>
#include <vector>
#include "boost/thread/mutex.hpp"
#include "boost/unordered_map.hpp"
#include "boost/unordered_set.hpp"
#include "caffe/internode/configuration.hpp"
#include "caffe/solver.hpp"

namespace caffe {

template <typename Dtype>
class BlobCodec;

template <typename Dtype>
class SynchronousParamServer : public internode::MultiWaypoint::Handler,
                               public internode::Waypoint::Handler {
  shared_ptr<internode::Daemon> daemon;
  shared_ptr<internode::MultiWaypoint> waypoint;
  shared_ptr<Solver<Dtype> > solver;

  typedef std::vector<std::vector<int> > BlobVersion;
  typedef boost::unordered_map<internode::RemoteId, BlobVersion> VersionMap;
  typedef boost::unordered_set<internode::RemoteId> ClientSet;

  ClientSet all_clients;
  VersionMap version_sent;
  vector<vector<int> > blob_version;
  vector<vector<int> > blob_iters;
  vector<vector<ClientSet> > pending_clients;
  shared_ptr<BlobCodec<Dtype> > codec;

  void init_client(internode::RemoteId);
 public:
  SynchronousParamServer(shared_ptr<Solver<Dtype> >,
                         string bind_address);
  void run();

  virtual void accepted(internode::RemoteId);
  virtual void disconnected(internode::RemoteId);
  virtual void received(char* data, size_t size, internode::RemoteId);

  virtual void update_clients();
  virtual void upgrade_layer(int layer_id);
  bool all_layers_synced() const;
  int current_iter() const;
};
}  // namespace caffe


#endif  // CAFFE_SYNCHRONOUSPARAMSERVER_HPP_

