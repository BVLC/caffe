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

#ifndef CAFFE_SERIALIZATION_BLOBCODEC_HPP_
#define CAFFE_SERIALIZATION_BLOBCODEC_HPP_

#include <boost/function.hpp>
#include <boost/optional.hpp>
#include <algorithm>
#include <numeric>
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Net;

struct BlobEncoding {
  enum What {
    PARAMS = 0,
    GRADS = 1
  };
};
typedef BlobEncoding::What BlobEncodingWhat;

template <typename Dtype>
class BlobCodec {
 public:
  typedef typename BlobEncoding::What What;
  static shared_ptr<BlobCodec> create_codec(
    const MultinodeParameter& param,
    bool ensure_is_single_threaded);

  virtual uint32_t encode(BlobUpdate* msg,
                          const Blob<Dtype>* src,
                          What what,
                          uint32_t part) const = 0;

  virtual bool decode(const BlobUpdate& update,
                      Blob<Dtype>* dest,
                      What what,
                      Dtype alpha,
                      Dtype beta) const = 0;

  virtual size_t max_elements_per_part() const = 0;
  virtual size_t packet_size() const = 0;
};

template <typename Dtype>
double check_sum(const Dtype* data, size_t size) {
  if (size == 0) return 0.0;
  if (size == 1) return *data;
  if (size == 2) return static_cast<double>(data[0]) + data[1];
  return check_sum(data, size / 2)
    + check_sum(data + size / 2, size - size / 2);
}

template <typename Dtype>
double check_sum(Blob<Dtype>* blob, BlobEncodingWhat what) {
  return check_sum(
    ((what == BlobEncoding::PARAMS) ?  blob->cpu_data() : blob->cpu_diff()),
    blob->count());
}

}  // namespace caffe

#endif  // CAFFE_SERIALIZATION_BLOBCODEC_HPP_

