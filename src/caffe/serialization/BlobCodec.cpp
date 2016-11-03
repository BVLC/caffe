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

#include <algorithm>
#include <cfloat>
#include <numeric>
#include "boost/make_shared.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/serialization/bitfield.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

namespace {

size_t get_max_header_size() {
  BlobUpdate update;
  update.mutable_info()->set_version(UINT_MAX);
  update.mutable_info()->set_layer_id(INT_MAX);
  update.mutable_info()->set_blob_id(INT_MAX);
  update.mutable_info()->set_part(INT_MAX);
  update.set_iters(INT_MAX);
  update.mutable_compression_param()->set_algo(COMPRESSION_AVERAGING);
  ThresholdCompressionConfig* config =
    update.mutable_compression_param()->mutable_threshold_param();
  config->set_threshold(FLT_MAX);
  config->set_size(INT_MAX);
  config->set_multiplier(FLT_MAX);
  config->set_size(INT_MAX);
  update.set_data("abcd", 4);
  return update.ByteSize();
}

template <typename Dtype>
void encode_simple(Dtype* data, BlobUpdate* msg, uint32_t size) {
  msg->set_data(data, size * sizeof(Dtype));
}

template <bool SingleThreaded, typename Dtype>
bool decode_simple(Dtype* dest,
                   uint32_t max_size,
                   uint32_t part_size,
                   const BlobUpdate& update,
                   Dtype alpha,
                   Dtype beta) {
  uint32_t encoded_elements = update.data().size() / sizeof(Dtype);
  if (max_size < encoded_elements) {
    LOG(ERROR) << "ignoring received data for layer: "
               << update.info().layer_id()
               << " because part is over destination blob: "
               << "(available elements: " << max_size << ", "
               << "encoded elements: " << encoded_elements << ")";
    return false;
  }

  Dtype* src =
    reinterpret_cast<Dtype*>(const_cast<char*>(update.data().c_str()));
  if ((alpha == 1.0) && (beta == 0.0)) {
    caffe_copy(encoded_elements, src, dest);
    return true;
  }
  if (SingleThreaded) {
    // to ensure, that this doesn't interrupt actual calculation
    // it is done in a naive way, without using threaded implementations
    // naiveness here improves overall performance
    for (int i = 0; i < encoded_elements; ++i) {
      dest[i] = src[i] * alpha + dest[i] * beta;
    }
  } else {
    // for server we might not care about multiple threads spawn by below
    caffe_cpu_axpby<Dtype>(encoded_elements, alpha, src, beta, dest);
  }
  return true;
}

template <typename Dtype>
void encode_averaging(Dtype* data, BlobUpdate* msg, uint32_t size) {
  ThresholdCompressionConfig& config =
    *msg->mutable_compression_param()->mutable_threshold_param();

  float sum = 0.0;
  for (int i = 0; i < size; ++i) {
    sum = sum + fabs(data[i]);
  }

  Dtype threshold = sum / Dtype(size) * config.multiplier();
  uint32_t max_val =
    static_cast<uint32_t>(floor(pow(2.0, config.size() - 1))) - 1;
  config.set_threshold(threshold);

  bitfield buffer(size * config.size());
  uint32_t bit = 0;

  for (int i = 0; i < size; ++i) {
    int32_t val =
      std::min(uint32_t(round(fabs(data[i]) / threshold)), max_val);
    if (data[i] < 0) val = -val;
    buffer.set_bits(val, config.size() - 1, bit);

    CHECK_EQ(val, buffer.get_bits(config.size() - 1, bit))
      << " actual:" << data[i] << " " << (fabs(data[i]) / threshold);

    bit += config.size();
  }
  msg->set_data(buffer.raw(), buffer.bytes());
}

template <typename Dtype>
bool decode_averaging(Dtype* dest,
                      int32_t max_size,
                      uint32_t part_size,
                      const BlobUpdate& msg,
                      Dtype alpha,
                      Dtype beta) {
  ThresholdCompressionConfig config =
    msg.compression_param().threshold_param();

  if (!config.has_size()) {
    LOG(ERROR) << "ignoring received data for layer: " << msg.info().layer_id()
               << " because of missing threshold";
    return false;
  }

  int blob_bytes = bitfield(max_size * config.size()).bytes();
  if (blob_bytes < msg.data().size()) {
    LOG(ERROR) << "ignoring received data for layer: " << msg.info().layer_id()
               << " and blob: " << msg.info().blob_id()
               << " because the received blob size is too big: "
               << msg.data().size() << " > " << blob_bytes;
    return false;
  }

  bitfield buffer(msg.data().c_str(), msg.data().size());
  uint32_t bit = 0;

  for (int j = 0; j < msg.data().size() / sizeof(Dtype); ++j) {
    int32_t mult = buffer.get_bits(config.size() - 1, bit);
    bit += config.size();
    Dtype val = config.threshold() * Dtype(mult);
    dest[j] = dest[j] * beta + val * alpha;
  }
  return true;
}

template <typename Dtype, bool SingleThreaded>
struct BlobCodecImpl : BlobCodec<Dtype> {
  MultinodeParameter param;
  const size_t max_header_size;
  const size_t max_packet_size;
  const size_t elements_per_part;

  explicit BlobCodecImpl(MultinodeParameter param)
    : param(param)
    , max_header_size(get_max_header_size())
    , max_packet_size(param.max_packet_size())
    , elements_per_part((max_packet_size - max_header_size) / sizeof(Dtype)) {
    CHECK(max_packet_size > (max_header_size + sizeof(Dtype)))
      << "packet size must accomodate for proto msg size, "
      << "min packet size must be greater than: "
      << (max_header_size + sizeof(Dtype));
  }

  virtual size_t max_elements_per_part() const {
    return elements_per_part;
  }

  virtual size_t packet_size() const {
    return max_packet_size;
  }

  virtual uint32_t encode(BlobUpdate* msg,
                          const Blob<Dtype>* src,
                          typename BlobCodec<Dtype>::What what,
                          uint32_t part) const {
    const uint32_t start_element = part * elements_per_part;
    CHECK(start_element < src->count());
    const uint32_t size =
      std::min(uint32_t(start_element + elements_per_part),
               uint32_t(src->count())) - start_element;
    msg->mutable_info()->set_part(part);
    *msg->mutable_compression_param() = param.outgoing_compression();

    const Dtype* data =
      ((what == BlobEncoding::GRADS) ?
        src->cpu_diff() : src->cpu_data()) + start_element;
    DLOG(INFO) << "encoding " <<
      ((what == BlobEncoding::GRADS) ? "grads" : "params")
      << ", number of elements: " << size
      << ", max_packet_size: " << max_packet_size
      << ", elements_per_part: " << elements_per_part
      << ", max_header_size: " << max_header_size
      << ", part: " << msg->info().part()
      << ", starting from: " << start_element
      << ", total size: " << src->count();

    if (param.outgoing_compression().algo() == COMPRESSION_AVERAGING) {
      encode_averaging(data, msg, size);
    } else {
      encode_simple(data, msg, size);
    }

    return size;
  }

  virtual bool decode(const BlobUpdate& update,
                      Blob<Dtype>* dest,
                      typename BlobCodec<Dtype>::What what,
                      Dtype alpha,
                      Dtype beta) const {
    if (update.data().size() % sizeof(Dtype) != 0) {
      LOG(ERROR) << "ignoring received data for layer: "
                 << update.info().layer_id()
                 << " because data is corrupted, data size is not divisable"
                 << " by size of element";
      return false;
    }

    Dtype* data =
      ((what == BlobEncoding::GRADS) ?
        dest->mutable_cpu_diff() : dest->mutable_cpu_data())
      + update.info().part() * elements_per_part;
    int32_t max_size = dest->count() - update.info().part() * elements_per_part;


    DLOG(INFO) << "decoding " <<
      ((what == BlobEncoding::GRADS) ? "grads" : "params")
      << ", number of elements: " << (update.data().size() / sizeof(Dtype))
      << ", part: " << update.info().part()
      << ", starting from: " << update.info().part() * elements_per_part
      << ", total size: " << dest->count();
    if (update.compression_param().algo() == COMPRESSION_AVERAGING) {
      return decode_averaging(
        data, max_size, elements_per_part, update, alpha, beta);
    }
    return decode_simple<SingleThreaded>(
        data, max_size, elements_per_part, update, alpha, beta);
  }
};

}  // namespace

template <typename Dtype>
shared_ptr<BlobCodec<Dtype> > BlobCodec<Dtype>::create_codec(
  const MultinodeParameter& param,
  bool single_threaded) {
  if (single_threaded)
    return boost::make_shared<BlobCodecImpl<Dtype, true> >(param);
  return boost::make_shared<BlobCodecImpl<Dtype, false> >(param);
}

INSTANTIATE_CLASS(BlobCodec);
}  // namespace caffe

