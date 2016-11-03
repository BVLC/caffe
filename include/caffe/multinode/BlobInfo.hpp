/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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

