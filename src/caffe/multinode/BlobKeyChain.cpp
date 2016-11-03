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

#include <vector>
#include "boost/make_shared.hpp"
#include "boost/thread.hpp"
#include "caffe/multinode/BlobKeyChain.hpp"

namespace caffe {


namespace {

template <typename Dtype, bool IsStub>
struct BlobKeyChainImpl : public BlobKeyChain<Dtype> {
  std::vector<shared_ptr<boost::recursive_mutex> > mtxs;

  explicit BlobKeyChainImpl(size_t layers)
    : mtxs(layers) {
    for (int i = 0; i < mtxs.size(); ++i) {
      mtxs[i].reset(new boost::recursive_mutex());
    }
  }

  virtual void lock(int layer_id) {
    CHECK_GE(layer_id, 0);
    CHECK(layer_id < mtxs.size());
    if (!IsStub) mtxs[layer_id]->lock();
  }
  virtual void unlock(int layer_id) {
    CHECK_GE(layer_id, 0);
    CHECK(layer_id < mtxs.size());
    if (!IsStub) mtxs[layer_id]->unlock();
  }

  virtual void lock(int layer_id, int blob_id, int part) {
    if (!IsStub) lock(layer_id);
  }
  virtual void unlock(int layer_id, int blob_id, int part) {
    if (!IsStub) unlock(layer_id);
  }
};
}  // namespace

template <typename Dtype>
shared_ptr<BlobKeyChain<Dtype> >
  BlobKeyChain<Dtype>::create(size_t layers) {
  return boost::make_shared<BlobKeyChainImpl<Dtype, false> >(layers);
}

template <typename Dtype>
shared_ptr<BlobKeyChain<Dtype> >
  BlobKeyChain<Dtype>::create_empty(size_t layers) {
  return boost::make_shared<BlobKeyChainImpl<Dtype, true> >(layers);
}

INSTANTIATE_CLASS(BlobKeyChain);
}  // namespace caffe

