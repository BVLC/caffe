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

  struct IterSizeHandler {
    virtual void received_iter_size(internode::RemoteId from, int iters) = 0;
  };

  static shared_ptr<BlobComms> create(
    shared_ptr<BlobAccessor<Dtype> > blob_accessor,
    shared_ptr<BlobConstInfo> const_info,
    shared_ptr<BlobSyncInfo> sync_info,
    shared_ptr<internode::Waypoint> waypoint,
    shared_ptr<BlobCodec<Dtype> > codec,
    shared_ptr<BlobKeyChain<Dtype> > keychain,
    Settings settings,
    int num_of_threads);

  virtual void push(int layer_id, uint32_t version) = 0;
  virtual void push(int layer_id, int blob_id, int part, uint32_t version) = 0;
  virtual void cancel(int layer_id, uint32_t version) = 0;
  virtual void received(char* data, size_t size, internode::Waypoint*) = 0;

  virtual void send_iter_size(int iter_size) = 0;
  virtual void register_iter_size_handler(IterSizeHandler* handler) = 0;
};

}  // namespace caffe

#endif  // CAFFE_MULTINODE_BLOBCOMMS_HPP_

