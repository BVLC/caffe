#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <boost/optional.hpp>
#include <boost/ref.hpp>
#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>
#include <algorithm>
#include <deque>
#include <vector>
#include "caffe/internode/communication.hpp"
#include "caffe/multinode/BlobComms.hpp"
#include "caffe/multinode/SendCallback.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/serialization/ProtoSerialize.hpp"

namespace caffe {

using internode::RemoteId;
using internode::Waypoint;

namespace {

struct Part {
  int layer_id;
  int blob_id;
  int part;
  uint32_t version;
};

template <typename Dtype>
struct BlobCommsImpl : BlobComms<Dtype> {
  const shared_ptr<Solver<Dtype> > solver;
  const shared_ptr<BlobConstInfo> const_info;
  const shared_ptr<BlobSyncInfo> sync_info;
  const shared_ptr<BlobKeyChain<Dtype> > keychain;
  const shared_ptr<internode::Waypoint> waypoint;
  const shared_ptr<BlobCodec<Dtype> > codec;
  const uint32_t iter_size;
  const typename BlobComms<Dtype>::Settings settings;

  char* buffer;

  vector<uint32_t> sending_version;
  vector<uint32_t> cancelled_version;
  std::vector<std::vector<Part> > all_parts;
  std::deque<Part> to_send;
  bool during_sending;

  BlobCommsImpl(shared_ptr<Solver<Dtype> > solver,
                shared_ptr<BlobInfo<Dtype> > info,
                shared_ptr<BlobKeyChain<Dtype> > keychain,
                shared_ptr<internode::Waypoint> waypoint,
                shared_ptr<BlobCodec<Dtype> > codec,
                uint32_t iter_size,
                typename BlobComms<Dtype>::Settings settings)
    : solver(solver)
    , const_info(info->get_const_info())
    , sync_info(info->get_sync_info())
    , keychain(keychain)
    , waypoint(waypoint)
    , codec(codec)
    , iter_size(iter_size)
    , settings(settings)
    , buffer(new char[codec->packet_size()])
    , sending_version(const_info->layers(), 0)
    , cancelled_version(const_info->layers(), 0)
    , during_sending(false) {
    for (int i = 0; i < const_info->layers(); ++i) {
      std::vector<Part> parts;
      for (int j = 0; j < const_info->blobs(i); ++j) {
        for (int k = 0; k < const_info->parts(i, j); ++k) {
          Part part = {i, j, k, 0u};
          parts.push_back(part);
        }
      }
      all_parts.push_back(parts);
    }
  }

  Blob<Dtype>* get_blob(int layer_id, int blob_id) {
    CHECK_GE(layer_id, 0);
    CHECK_GE(blob_id, 0);
    CHECK(layer_id < solver->net()->layers().size());
    CHECK(blob_id < solver->net()->layers()[layer_id]->blobs().size());
    return solver->net()->layers()[layer_id]->blobs()[blob_id].get();
  }
  template <typename PartInfo>
  Blob<Dtype>* get_blob(const PartInfo& item) {
    return get_blob(item.layer_id, item.blob_id);
  }

  boost::optional<Part> get_next_part_to_send() {
    while (!to_send.empty()) {
      Part ret = to_send.front();
      to_send.pop_front();
      if (sending_version[ret.layer_id] > cancelled_version[ret.layer_id]) {
        return ret;
      }
    }
    return boost::none;
  }

  void send() {
    if (during_sending) {
      DLOG(INFO) << "during_sending";
      return;
    }
    boost::optional<Part> next = get_next_part_to_send();
    if (!next) return;

    BlobUpdate update;
    update.mutable_info()->set_layer_id(next->layer_id);
    update.mutable_info()->set_blob_id(next->blob_id);
    update.mutable_info()->set_part(next->part);
    update.mutable_info()->set_version(sending_version[next->layer_id]);
    update.set_iters(iter_size);
    DLOG(INFO) << "sending update of layer " << update.info().layer_id()
      << ", blob " << update.info().blob_id()
      << ", part " << update.info().part()
      << " of version: " << update.info().version();

    keychain->lock(update.info().layer_id());
    codec->encode(
      &update, get_blob(*next), settings.what_sent, update.info().part());
    keychain->unlock(update.info().layer_id());

    update.SerializeToArray(buffer, codec->packet_size());
    during_sending = true;
    waypoint->async_send(
      buffer, update.ByteSize(), boost::bind(&BlobCommsImpl::sent, this));
    DLOG(INFO) << "sent update of layer " << update.info().layer_id()
      << ", blob " << update.info().blob_id()
      << ", part " << update.info().part()
      << " of version: " << update.info().version();
  }

  void sent() {
    during_sending = false;
    send();
  }

  virtual uint32_t currently_sending_version() const {
    uint32_t ret = 0;
    for (int i = 0; i < sending_version.size(); ++i) {
      ret = std::max(ret, sending_version[i]);
    }
    return ret;
  }

  virtual uint32_t currently_sending_version(int layer_id) const {
    CHECK_GE(layer_id, 0);
    CHECK(layer_id < sending_version.size());
    return sending_version[layer_id];
  }

  void push(int layer_id, uint32_t version) {
    sending_version[layer_id] = std::max(version, sending_version[layer_id]);
    to_send.insert(
      to_send.begin(), all_parts[layer_id].begin(), all_parts[layer_id].end());
    DLOG(INFO) << "pushed: " << layer_id << " with version " << version
      << " to_send.size(): " << to_send.size();
    send();
  }

  void cancel(int layer_id, uint32_t version) {
    cancelled_version[layer_id] = version;
  }

  virtual void received(char* data,
                        size_t size,
                        internode::Waypoint* waypoint) {
    BlobUpdate msg;
    if (!deserialize(data, size, &msg)) {
      LOG(ERROR) << "deserialize failed";
      return;
    }

    if (!msg.has_info()) {
      LOG(ERROR) << "msg has no info";
      return;
    }

    Blob<Dtype>* blob = get_blob(msg.info().layer_id(), msg.info().blob_id());
    if (sync_info->received_version(
          waypoint->id(),
          msg.info().layer_id(), msg.info().blob_id(), msg.info().part())
        >= msg.info().version()) {
      DLOG(INFO) << "ignoring old blob update for blob: "
            << msg.info().blob_id()
            << " of layer " << msg.info().layer_id()
            << ", blob: " << msg.info().blob_id()
            << ", part: " << msg.info().part()
            << " with version " << msg.info().version();
      return;
    }

    DLOG(INFO) << "received update for blob: " << msg.info().blob_id()
               << " of layer " << msg.info().layer_id()
               << ", part " << msg.info().part()
               << " with version " << msg.info().version()
               << " current version: "
               << sync_info->received_version(
                    waypoint->id(), msg.info().layer_id(), msg.info().blob_id(),
                    msg.info().part())
               << " data size: " << msg.data().size();

    keychain->lock(msg.info().layer_id());
    if (!codec->decode(
      msg, blob,
      settings.what_received,
      settings.received_incoming_multiplier,
      settings.received_current_multiplier)) {
      LOG(ERROR) << "decoding failed";
      return;
    }
    if (Caffe::mode() == Caffe::GPU) {
      blob->gpu_data();
    }
    keychain->unlock(msg.info().layer_id());

    sync_info->received(
      waypoint->id(), msg.info().layer_id(), msg.info().blob_id(),
      msg.info().part(), msg.info().version(), msg.iters());
  }
};

}  // namespace

template <typename Dtype>
shared_ptr<BlobComms<Dtype> > BlobComms<Dtype>::create(
    shared_ptr<Solver<Dtype> > solver,
    shared_ptr<BlobInfo<Dtype> > info,
    shared_ptr<BlobKeyChain<Dtype> > keychain,
    shared_ptr<internode::Waypoint> waypoint,
    shared_ptr<BlobCodec<Dtype> > codec,
    Settings settings) {
  return boost::make_shared<BlobCommsImpl<Dtype> >(
      solver, info, keychain, waypoint, codec,
      solver->param().iter_size(), settings);
}

template <typename Dtype>
BlobComms<Dtype>::Settings::Settings(
  BlobEncodingWhat what_sent,
  BlobEncodingWhat what_received,
  Dtype received_incoming_multiplier,
  Dtype received_current_multiplier)
  : what_sent(what_sent)
  , what_received(what_received)
  , received_incoming_multiplier(received_incoming_multiplier)
  , received_current_multiplier(received_current_multiplier) {
}

template <typename Dtype>
BlobComms<Dtype>::Settings::Settings(const Settings& other)
  : what_sent(other.what_sent)
  , what_received(other.what_received)
  , received_incoming_multiplier(other.received_incoming_multiplier)
  , received_current_multiplier(other.received_current_multiplier) {
}

INSTANTIATE_CLASS(BlobComms);

}  // namespace caffe

