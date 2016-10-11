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

