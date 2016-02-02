#ifndef CAFFE_MULTINODE_BLOBCOMMS_HPP_
#define CAFFE_MULTINODE_BLOBCOMMS_HPP_

#include "caffe/internode/communication.hpp"
#include "caffe/multinode/BlobInfo.hpp"
#include "caffe/multinode/BlobKeyChain.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/solver.hpp"

namespace caffe {

template <typename Dtype>
class BlobComms : public internode::Waypoint::Handler {
 public:
  class Settings {
   public:
    BlobEncodingWhat what_sent;
    BlobEncodingWhat what_received;
    Dtype received_incoming_multiplier;
    Dtype received_current_multiplier;
    Settings(BlobEncodingWhat what_sent,
             BlobEncodingWhat what_received,
             Dtype received_incoming_multiplier,
             Dtype received_current_multiplier);
    Settings(const Settings& other);
  };

  static shared_ptr<BlobComms> create(
    shared_ptr<Solver<Dtype> > solver,
    shared_ptr<BlobInfo<Dtype> > info,
    shared_ptr<BlobKeyChain<Dtype> > keychain,
    shared_ptr<internode::Waypoint> waypoint,
    shared_ptr<BlobCodec<Dtype> > codec,
    Settings settings);

  virtual uint32_t currently_sending_version() const = 0;
  virtual uint32_t currently_sending_version(int layer_id) const = 0;
  virtual void push(int layer_id, uint32_t version) = 0;
  virtual void cancel(int layer_id, uint32_t version) = 0;
  virtual void received(char* data, size_t size, internode::Waypoint*) = 0;
};

}  // namespace caffe

#endif  // CAFFE_MULTINODE_BLOBCOMMS_HPP_

