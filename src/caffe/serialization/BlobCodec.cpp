#include <algorithm>
#include <numeric>
#include "boost/make_shared.hpp"
#include "caffe/serialization/bitfield.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

namespace {


template <typename Dtype>
void encode_simple(Dtype* data, BlobUpdate* msg, uint32_t size) {
  msg->set_data(data, size * sizeof(Dtype));
}

template <typename Dtype>
bool decode_simple(Dtype* dest,
                   uint32_t max_size,
                   uint32_t part_size,
                   const BlobUpdate& update,
                   Dtype alpha,
                   Dtype beta) {
  uint32_t encoded_elements = update.data().size() / sizeof(Dtype);
  if (max_size < update.part() * part_size + encoded_elements) {
    LOG(ERROR) << "ignoring received data for layer: " << update.layer_id()
               << " because part is over destination blob: "
               << "(available data size: " << max_size << ", "
               << "last data element: "
               << update.part() * part_size + encoded_elements << ")";
    return false;
  }

  Dtype* src =
    reinterpret_cast<Dtype*>(const_cast<char*>(update.data().c_str()));

  // to ensure, that this doesn't interrupt actual calculation
  // it is done in a naive way, without using threaded implementations
  // naiveness here improves overall performance
  for (int i = 0; i < encoded_elements; ++i) {
    dest[i] = src[i] * alpha + dest[i] * beta;
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
    LOG(ERROR) << "ignoring received data for layer: " << msg.layer_id()
               << " because of missing threshold";
    return false;
  }

  int blob_bytes = bitfield(max_size * config.size()).bytes();
  if (blob_bytes < msg.data().size()) {
    LOG(ERROR) << "ignoring received data for layer: " << msg.layer_id()
               << " and blob: " << msg.blob_id()
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

template <typename Dtype>
struct BlobCodecImpl : BlobCodec<Dtype> {
  MultinodeParameter param;

  explicit BlobCodecImpl(MultinodeParameter param) : param(param) {
  }

  virtual uint32_t encode(BlobUpdate* msg,
                          const Blob<Dtype>* src,
                          typename BlobCodec<Dtype>::What what,
                          uint32_t start_element) const {
    uint32_t part_size = param.max_packet_size() / sizeof(Dtype);
    uint32_t size =
      std::min(start_element + part_size, uint32_t(src->count()))
        - start_element;
    msg->set_part(start_element / part_size);
    *msg->mutable_compression_param() = param.outgoing_compression();

    const Dtype* data =
      ((what == BlobCodec<Dtype>::GRADS) ?
        src->cpu_diff() : src->cpu_data()) + start_element;
    VLOG(5) << "encoding " <<
      ((what == BlobCodec<Dtype>::GRADS) ? "grads" : "params")
      << ", number of elements: " << size
      << ", part: " << msg->part()
      << ", starting from: " << start_element;

    if (param.outgoing_compression().algo() == AVERAGING) {
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
      LOG(ERROR) << "ignoring received data for layer: " << update.layer_id()
                 << " because data is corrupted, data size is not divisable"
                 << " by size of element";
      return false;
    }

    uint32_t part_size = param.max_packet_size() / sizeof(Dtype);
    Dtype* data =
      ((what == BlobCodec<Dtype>::GRADS) ?
        dest->mutable_cpu_diff() : dest->mutable_cpu_data())
      + update.part() * part_size;
    int32_t max_size = dest->count() - update.part() * part_size;


    VLOG(5) << "decoding " <<
      ((what == BlobCodec<Dtype>::GRADS) ? "grads" : "params")
      << ", number of elements: " << (update.data().size() / sizeof(Dtype))
      << ", part: " << update.part()
      << ", starting from: " << update.part() * part_size;
    if (update.compression_param().algo() == AVERAGING) {
      return decode_averaging(data, max_size, part_size, update, alpha, beta);
    }
    return decode_simple(data, max_size, part_size, update, alpha, beta);
  }
};

}  // namespace

template <typename Dtype>
shared_ptr<BlobCodec<Dtype> > BlobCodec<Dtype>::create_codec(
  const MultinodeParameter& param) {
  return boost::make_shared<BlobCodecImpl<Dtype> >(param);
}

INSTANTIATE_CLASS(BlobCodec);
}  // namespace caffe

