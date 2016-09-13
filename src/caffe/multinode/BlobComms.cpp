#include <boost/bind.hpp>
#include <boost/condition_variable.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/make_shared.hpp>
#include <boost/optional.hpp>
#include <boost/ref.hpp>
#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>
#include <algorithm>
#include <deque>
#include <vector>
#include "caffe/internode/communication.hpp"
#include "caffe/internode/tree_cluster.hpp"
#include "caffe/multinode/BlobComms.hpp"
#include "caffe/multinode/SendCallback.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/serialization/ProtoSerialize.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/math_functions.hpp"

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

template <typename Dtype, bool UseThreads>
struct BlobCommsImpl : BlobComms<Dtype> {
  const shared_ptr<BlobAccessor<Dtype> > blob_accessor;
  const shared_ptr<BlobConstInfo> const_info;
  const shared_ptr<BlobSyncInfo> sync_info;
  const shared_ptr<internode::Waypoint> waypoint;
  const shared_ptr<BlobCodec<Dtype> > codec;
  const shared_ptr<BlobKeyChain<Dtype> > keychain;
  const typename BlobComms<Dtype>::Settings settings;

  typedef typename BlobComms<Dtype>::IterSizeHandler IterSizeHandler;
  vector<IterSizeHandler*> iter_size_handlers;

  char* buffer;

  struct Worker {
    struct Job : Element {
      std::vector<char> buffer;
      size_t size;
      RemoteId id;
    };
    struct SendJob : Element {
    };
    BlockingQueue<Element*> jobs_to_run;
    boost::condition_variable cv;
    boost::mutex mtx;
    std::vector<Job*> available_jobs;
    boost::thread thread;
    BlobCommsImpl* impl;
    boost::shared_ptr<SendJob> send_job;

    Worker(BlobCommsImpl* impl, size_t packet_size)
      : thread(boost::bind(&Worker::execute, this))
      , impl(impl)
      , send_job(new SendJob()) {
    }

    ~Worker() {
      push_job(NULL, 0, 0);
      thread.join();
    }

    void push_job(char* data, size_t size, RemoteId id) {
      Job* job;
      {
        boost::mutex::scoped_lock lock(mtx);
        if (available_jobs.empty()) {
          job = new Job();
        } else {
          job = available_jobs.back();
          available_jobs.pop_back();
        }
      }
      if (job->buffer.size() < size) job->buffer.resize(size);
      caffe_copy(size, data, &job->buffer.front());
      job->size = size;
      job->id = id;
      jobs_to_run.push(job);
    }

    void push_send_job() {
      jobs_to_run.push(send_job.get());
    }

    void execute() {
      DLOG(INFO) << "Worker started " << this;
      while (true) {
        Element* elem = jobs_to_run.pop();
        if (elem == send_job.get()) {
          impl->send();
          continue;
        }
        Job* job = elem->cast<Job>();
        if ((job->size == 0) && (job->id == 0)) {
          break;
        }
        impl->handle(&job->buffer.front(), job->size, job->id);
        {
          boost::mutex::scoped_lock lock(mtx);
          available_jobs.push_back(job);
        }
      }
      DLOG(INFO) << "Worker finished " << this;
    }
  };

  mutable boost::recursive_mutex worker_mtx;
  mutable boost::recursive_mutex mtx;
  uint32_t worker;

  vector<uint32_t> sending_version;
  vector<uint32_t> cancelled_version;
  std::vector<std::vector<Part> > all_parts;
  std::deque<Part> to_send;
  bool during_sending;

  std::vector<boost::shared_ptr<Worker> > all_workers;

  BlobCommsImpl(shared_ptr<BlobAccessor<Dtype> > blob_accessor,
                shared_ptr<BlobConstInfo> const_info,
                shared_ptr<BlobSyncInfo> sync_info,
                shared_ptr<internode::Waypoint> waypoint,
                shared_ptr<BlobCodec<Dtype> > codec,
                shared_ptr<BlobKeyChain<Dtype> > keychain,
                typename BlobComms<Dtype>::Settings settings,
                uint32_t threads)
    : blob_accessor(blob_accessor)
    , const_info(const_info)
    , sync_info(sync_info)
    , waypoint(waypoint)
    , codec(codec)
    , keychain(keychain)
    , settings(settings)
    , buffer(new char[codec->packet_size()])
    , worker(0)
    , sending_version(const_info->layers(), 0)
    , cancelled_version(const_info->layers(), 0)
    , during_sending(false)
    , all_workers(threads) {
    for (int i = 0; i < const_info->layers(); ++i) {
      std::vector<Part> parts;
      for (int j = 0; j < const_info->blobs(i); ++j) {
        for (int k = 0; k < const_info->parts(i, j); ++k) {
          Part part = {i, j, k, 0u};
          parts.push_back(part);
        }
      }
      all_parts.push_back(parts);
      DLOG(INFO) << "parts[=" << i << "]=" << parts.size();
    }
    DLOG(INFO) << "all_parts_size=" << all_parts.size();
    for (int i = 0; i < threads; ++i)
      all_workers[i].reset(new Worker(this, codec->packet_size()));
  }

  Worker* get_worker() {
    boost::recursive_mutex::scoped_lock lock(worker_mtx);
    Worker* ret = all_workers[worker].get();
    worker = (worker + 1) % all_workers.size();
    return ret;
  }

  Blob<Dtype>* get_blob(int layer_id, int blob_id) {
    return blob_accessor->get_blob(layer_id, blob_id);
  }
  template <typename PartInfo>
  Blob<Dtype>* get_blob(const PartInfo& item) {
    return blob_accessor->get_blob(item.layer_id, item.blob_id);
  }

  boost::optional<Part> get_next_part_to_send() {
    boost::recursive_mutex::scoped_lock lock(mtx);
    while (!to_send.empty()) {
      Part ret = to_send.front();
      to_send.pop_front();
      DLOG(INFO) << "sendv[" << ret.layer_id
                 << "]=" << sending_version[ret.layer_id]
                 << ", cancelv[" << ret.layer_id << "]="
                 << cancelled_version[ret.layer_id];
      if (sending_version.at(ret.layer_id) >
          cancelled_version.at(ret.layer_id)) {
        return ret;
      } else {
        DLOG(INFO) << "discard send";
      }
    }
    return boost::none;
  }

  bool is_during_sending() const {
    boost::recursive_mutex::scoped_lock lock(mtx);
    return during_sending;
  }

  void send() {
    boost::optional<Part> next = boost::none;
    BlobUpdate update;
    {
      boost::recursive_mutex::scoped_lock lock(mtx);
      if (is_during_sending()) {
        DLOG(INFO) << "during_sending";
        return;
      }
      next = get_next_part_to_send();
      if (!next) {
        DLOG(INFO) << "nothing to send";
        finished=true;
        boost::notify_all(cv, lock);
        return;
      }
      during_sending = true;
    }

    update.mutable_info()->set_layer_id(next->layer_id);
    update.mutable_info()->set_blob_id(next->blob_id);
    update.mutable_info()->set_part(next->part);
    update.mutable_info()->set_version(sending_version[next->layer_id]);
    DLOG(INFO) << "sending update of layer " << update.info().layer_id()
      << ", blob " << update.info().blob_id()
      << ", part " << update.info().part()
      << " of version: " << update.info().version();

    keychain->lock(next->layer_id);
    codec->encode(
      &update, get_blob(*next), settings.what_sent, update.info().part());
    update.SerializeToArray(buffer, codec->packet_size());
    keychain->unlock(next->layer_id);

    waypoint->async_send(
      buffer, update.ByteSize(), boost::bind(&BlobCommsImpl::sent, this));
    DLOG(INFO) << "sent update of layer " << update.info().layer_id()
      << ", blob " << update.info().blob_id()
      << ", part " << update.info().part()
      << " of version: " << update.info().version()
      << " size: " << update.ByteSize();
  }

  void sent() {
    {
      boost::recursive_mutex::scoped_lock lock(mtx);
      during_sending = false;
    }
    if (UseThreads) {
      get_worker()->push_send_job();
    } else {
      send();
    }
  }

  void push(int layer_id, uint32_t version) {
    {
      CHECK_GE(layer_id, 0);
      CHECK_LT(layer_id, const_info->layers());
      boost::recursive_mutex::scoped_lock lock(mtx);
      sending_version[layer_id] = std::max(version, sending_version[layer_id]);
      to_send.insert(
        to_send.begin(),
        all_parts[layer_id].begin(),
        all_parts[layer_id].end());
      DLOG(INFO) << "pushed: " << layer_id << " with version " << version
        << " to_send.size(): " << to_send.size();
    }
    if (UseThreads) {
      get_worker()->push_send_job();
    } else {
      send();
    }
  }

  void push(int layer_id, int blob_id, int part_id, uint32_t version) {
    {
      CHECK_GE(layer_id, 0);
      CHECK_LT(layer_id, const_info->layers());
      CHECK_GE(blob_id, 0);
      CHECK_LT(blob_id, const_info->blobs(layer_id));
      CHECK_GE(part_id, 0);
      CHECK_LT(part_id, const_info->parts(layer_id, blob_id));
      boost::recursive_mutex::scoped_lock lock(mtx);
      sending_version[layer_id] = std::max(version, sending_version[layer_id]);
      Part part = {layer_id, blob_id, part_id, version};
      to_send.push_front(part);
      DLOG(INFO) << "pushed: "
        << "(" << layer_id << ", " << blob_id << ", " << part_id << ")"
        << " with version " << version
        << " to_send.size(): " << to_send.size();
    }
    if (UseThreads) {
      get_worker()->push_send_job();
    } else {
      send();
    }
  }

  void cancel(int layer_id, uint32_t version) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    cancelled_version[layer_id] = version;
  }

  virtual void received(char* data, size_t size,
                        internode::Waypoint* waypoint) {
    if (UseThreads) {
      get_worker()->push_job(data, size, waypoint->id());
    } else {
      handle(data, size, waypoint->id());
    }
  }

  void handle(char* data, size_t size, RemoteId id) {
    BlobUpdate msg;
    if (!deserialize(data, size, &msg)) {
      LOG(ERROR) << "deserialize failed";
      return;
    }

    if (!msg.has_info()) {
      if (msg.has_iters()) {
        DLOG(INFO) << "received iters: " << msg.iters() << " from " << id;
        vector<IterSizeHandler*> to_call;
        {
          boost::recursive_mutex::scoped_lock lock(mtx);
          to_call = iter_size_handlers;
        }
        for (int i = 0; i < to_call.size(); ++i) {
          to_call[i]->received_iter_size(id, msg.iters());
        }
      } else {
        LOG(ERROR) << "empty update blob message";
      }
      return;
    }

    // expected to be thread safe
    if (sync_info->received_version(
          id, msg.info().layer_id(), msg.info().blob_id(), msg.info().part())
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
               << "/" << const_info->parts(msg.info().layer_id(),
                                           msg.info().blob_id())
               << " with version " << msg.info().version()
               << " current version: "
               << sync_info->received_version(
                    id, msg.info().layer_id(), msg.info().blob_id(),
                    msg.info().part())
               << " data size: " << msg.data().size();

    Blob<Dtype>* blob = get_blob(msg.info().layer_id(), msg.info().blob_id());
    keychain->lock(msg.info().layer_id());
    bool result = codec->decode(msg,
                                blob,
                                settings.what_received,
                                settings.received_incoming_multiplier,
                                settings.received_current_multiplier);
    keychain->unlock(msg.info().layer_id());
    if (!result) {
      LOG(ERROR) << "decoding failed";
      return;
    }

    sync_info->received(
      id, msg.info().layer_id(), msg.info().blob_id(),
      msg.info().part(), msg.info().version());
  }

  void send_iter_size(int iter_size) {
    BlobUpdate update;
    update.set_iters(iter_size);
    while (true) {
      boost::recursive_mutex::scoped_lock lock(mtx);
      if (is_during_sending()) continue;
      during_sending = true;
      break;
    }

    VLOG(2) << "sending iter size info (iter_size: " << iter_size << ")";

    update.SerializeToArray(buffer, codec->packet_size());

    waypoint->async_send(
      buffer, update.ByteSize(), boost::bind(&BlobCommsImpl::sent, this));
  }

  void register_iter_size_handler(IterSizeHandler* handler) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    iter_size_handlers.push_back(handler);
  }

  void finish_all_tasks() {
    boost::recursive_mutex::scoped_lock lock(mtx);
    while (!finished) {
      cv.wait(lock);
    }
  }
};

}  // namespace

template <typename Dtype>
shared_ptr<BlobComms<Dtype> > BlobComms<Dtype>::create(
    shared_ptr<BlobAccessor<Dtype> > blob_accessor,
    shared_ptr<BlobConstInfo> const_info,
    shared_ptr<BlobSyncInfo> sync_info,
    shared_ptr<internode::Waypoint> waypoint,
    shared_ptr<BlobCodec<Dtype> > codec,
    shared_ptr<BlobKeyChain<Dtype> > keychain,
    Settings settings,
    int num_of_threads) {

  if (num_of_threads < 2) {
    return boost::make_shared<BlobCommsImpl<Dtype, false> >(blob_accessor,
      const_info, sync_info, waypoint, codec, keychain, settings, 0);
  }
  return boost::make_shared<BlobCommsImpl<Dtype, true> >(blob_accessor,
      const_info, sync_info, waypoint, codec, keychain, settings,
      num_of_threads);
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

