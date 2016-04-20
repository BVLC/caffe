#ifndef CAFFE_MULTINODE_BLOBSYNCINFO_HPP_
#define CAFFE_MULTINODE_BLOBSYNCINFO_HPP_

#include "caffe/internode/communication.hpp"
#include "caffe/multinode/BlobInfo.hpp"
#include "caffe/multinode/BlobKeyChain.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/solver.hpp"

namespace caffe {

class BlobConstInfo {
 public:
  virtual uint32_t parts(int layer_id, int blob_id) const = 0;
  virtual uint32_t parts(int layer_id) const = 0;
  virtual uint32_t parts() const = 0;
  virtual uint32_t blobs(int layer_id) const = 0;
  virtual uint32_t layers() const = 0;
  virtual bool needs_syncing(int layer_id) const = 0;

  virtual ~BlobConstInfo() {}
};

class Sync {
 public:
  virtual bool received(internode::RemoteId from,
                        int layer_id,
                        int blob_id,
                        int part,
                        uint32_t version) = 0;
  virtual ~Sync() {}
};

class BlobSyncInfo : public Sync {
 public:
  virtual bool received(internode::RemoteId from,
                        int layer_id,
                        int blob_id,
                        int part,
                        uint32_t version) = 0;

  struct Handler {
    virtual void synced(int layer_id,
                        int blob_id,
                        int part,
                        uint32_t version) = 0;
    virtual void synced(int layer_id, uint32_t version) = 0;
    virtual void synced(uint32_t version) = 0;
  };
  virtual void register_synced_handler(Handler* handler) = 0;

  virtual uint32_t received_version(
    internode::RemoteId from, int layer_id, int blob_id, int part) const = 0;

  virtual void add_remote(internode::RemoteId id) = 0;
  virtual void remove_remote(internode::RemoteId id) = 0;
};

template <typename Dtype>
class BlobAccessor {
 public:
  virtual Blob<Dtype>* get_blob(int layer_id, int blob_id) = 0;
};

template <typename Dtype>
class BlobInfoFactory {
 public:
  static shared_ptr<BlobConstInfo> create_const_info(
    shared_ptr<Solver<Dtype> > solver,
    size_t elements_per_packet);

  static shared_ptr<BlobSyncInfo>  create_sync_info(
    shared_ptr<BlobConstInfo> const_info);

  static shared_ptr<BlobAccessor<Dtype> > create_blob_accessor(
    shared_ptr<Solver<Dtype> > solver);
};

}  // namespace caffe

#endif  // CAFFE_MULTINODE_BLOBSYNCINFO_HPP_

