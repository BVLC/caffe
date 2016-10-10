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

#include <boost/make_shared.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/unordered_map.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <stdint.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "caffe/internode/mpi_configuration.hpp"
#include "caffe/layers/remote_data_layer.hpp"
#include "caffe/multinode/SendCallback.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

using internode::RemoteId;
using internode::Waypoint;
using internode::Daemon;

template <typename Dtype>
void RemoteDataLayer<Dtype>::prepare(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  RemoteDataParameter param = Layer<Dtype>::layer_param().remote_data_param();
  if (param.shape_size() > 0) {
    CHECK(top.size() == param.shape_size());
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(param.shape(i));
    }
    CHECK_EQ(bottom.size(), 0);
  } else {
    CHECK(aux_blobs.size() == top.size());
    aux_blobs[1] = top[1];
    queue->get("Remote data layer prefetch queue empty", aux_blobs);
    this->data_transformer_->Transform(transform_blob.get(), top[0]);
  }

  CHECK(bottom.empty());
}

namespace {

template <typename Dtype>
struct Data : Element {
  vector<Blob<Dtype>*> top;
};

}  // namespace

template <typename Dtype>
struct RemoteDataLayer<Dtype>::RemoteDataQueue {
  virtual bool add_data(DataMsg* update) = 0;
  virtual void get(string msg, const vector<Blob<Dtype>*>& ret) = 0;
};

namespace {

template <typename Dtype>
struct ReusingShufflingDataCache : RemoteDataLayer<Dtype>::RemoteDataQueue {
  vector<Data<Dtype> > cache;
  Data<Dtype> batch;
  int read;
  int batch_size;
  int cache_size;
  bool read_all;
  boost::shared_ptr<Caffe::RNG> rng;
  boost::mutex mtx;

  template <typename ShapeInfo>
  void create_cache(ShapeInfo info) {
    batch.top.resize(info.shape_size());
    CHECK_GT(info.shape().size(), 0);
    for (int j = 0; j < info.shape_size(); ++j) {
      batch.top[j] = new Blob<Dtype>();
      batch.top[j]->Reshape(info.shape(j));
    }
    batch_size = batch.top[0]->shape()[0];
    for (int j = 0; j < info.shape_size(); ++j) {
      CHECK(batch.top[j]->shape()[0] == batch_size);
    }

    for (int i = 0; i < batch_size * cache_size; ++i) {
      Data<Dtype> data;
      data.top.resize(batch.top.size());

      for (int j = 0; j < data.top.size(); ++j) {
        vector<int> shape = batch.top[j]->shape();
        shape[0] = 1;
        data.top[j] = new Blob<Dtype>();
        data.top[j]->Reshape(shape);
      }
      cache.push_back(data);
    }
  }

  explicit ReusingShufflingDataCache(RemoteDataParameter param)
    : read(0)
    , batch_size(0)
    , cache_size(param.cache_size())
    , read_all(false)
    , rng(new Caffe::RNG(caffe_rng_rand())) {
    if (param.shape_size() > 0) {
      create_cache(param);
    }
  }

  virtual bool add_data(DataMsg* update) {
    int curr_read = 0;
    {
      boost::mutex::scoped_lock lock(mtx);
      curr_read = read;

      if (cache.empty()) {
        create_cache(*update);
      }
    }

    CHECK_GT(cache.size(), 0);
    CHECK(cache[0].top.size() == update->data_size());
    CHECK(batch.top.size() == update->data_size());

    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < update->data_size(); ++j) {
        Blob<Dtype>* target = cache[curr_read].top[j];

        int offset = batch.top[j]->offset(i);
        Dtype* src = reinterpret_cast<Dtype*>(
                   const_cast<char*>(update->data(j).c_str()));
        caffe_copy(target->count(), src + offset, target->mutable_cpu_data());
      }

      ++curr_read;
      if (curr_read == cache.size()) {
        break;
      }
    }

    {
      boost::mutex::scoped_lock lock(mtx);
      read = curr_read;
      read_all = (read == cache.size());
      LOG(INFO) << "added data " << read
                << "/" << cache.size() << " " << read_all;
    }
    return (curr_read < cache.size());
  }

  virtual void get(string msg, const vector<Blob<Dtype>*>& ret) {
    while (true) {
      boost::mutex::scoped_lock lock(mtx);
      if (read_all) {
        break;
      }
    }

    boost::uniform_int<int> dist(0, cache.size() - 1);

    CHECK_GT(batch_size, 0);
    for (int j = 0; j < ret.size(); ++j) {
      ret[j]->ReshapeLike(*batch.top[j]);
    }
    for (int i = 0; i < batch_size; ++i) {
      int curr = dist(*static_cast<caffe::rng_t*>(rng->generator()));
      for (int j = 0; j < ret.size(); ++j) {
        Blob<Dtype>* target = ret[j];
        Blob<Dtype>* src = cache[curr].top[j];

        int offset = batch.top[j]->offset(i);
        caffe_copy(src->count(),
                   src->cpu_data(),
                   target->mutable_cpu_data() + offset);
      }
    }
  }
};

template <typename Dtype>
struct SimpleDataCache : RemoteDataLayer<Dtype>::RemoteDataQueue {
  BlockingQueue<Element*> cached;
  BlockingQueue<Element*> waiting;
  BlockingQueue<Element*> free;
  int cache_size;

  template <typename ShapeInfo>
  void create_cache(ShapeInfo info) {
    for (int i = 0; i < cache_size; ++i) {
      Data<Dtype>* data = new Data<Dtype>();
      data->top.resize(info.shape_size());

      for (int j = 0; j < data->top.size(); ++j) {
        data->top[j] = new Blob<Dtype>();
        data->top[j]->Reshape(info.shape(j));
      }

      free.push(data);
    }

    if (cache_size == 0) {
      Data<Dtype>* data = new Data<Dtype>();
      data->top.resize(info.shape_size());

      for (int j = 0; j < data->top.size(); ++j) {
        data->top[j] = new Blob<Dtype>();
        data->top[j]->Reshape(info.shape(j));
      }

      waiting.push(data);
    }
  }

  explicit SimpleDataCache(RemoteDataParameter param)
    : cache_size(param.cache_size()) {

    if (param.shape_size() > 0) {
      create_cache(param);
    }
  }

  virtual bool add_data(DataMsg* update) {
    if (cached.size() + free.size() + waiting.size() == 0) {
      create_cache(*update);
    }

    Data<Dtype>* data = static_cast<Data<Dtype>*>(free.pop(""));

    CHECK(data->top.size() == update->data_size());
    for (int j = 0; j < data->top.size(); ++j) {
      CHECK(data->top[j]->count() * sizeof(Dtype) == update->data(j).size());
      caffe_copy(data->top[j]->count(),
                 reinterpret_cast<Dtype*>(
                   const_cast<char*>(update->data(j).c_str())),
                 data->top[j]->mutable_cpu_data());
    }

    cached.push(data);
    return true;
  }

  virtual void get(string msg, const vector<Blob<Dtype>*>& ret) {
    if (cache_size == 0) {
      Data<Dtype>* data = waiting.pop(msg + " waiting")
        ->template cast<Data<Dtype> >();
      free.push(data);
    }
    Data<Dtype>* data = cached.pop(msg)->template cast<Data<Dtype> >();
    CHECK(data->top.size() == ret.size());
    for (int i = 0; i < ret.size(); ++i) {
      ret[i]->ReshapeLike(*data->top[i]);
      caffe_copy(data->top[i]->count(), data->top[i]->cpu_data(),
                 ret[i]->mutable_cpu_data());
    }

    if (cache_size > 0) {
      free.push(data);
    } else {
      waiting.push(data);
    }
  }
};

template <typename Dtype>
struct RemoteDataReader : InternalThread, Waypoint::Handler {
  typedef typename RemoteDataLayer<Dtype>::RemoteDataQueue Queue;

  shared_ptr<Daemon> daemon;
  shared_ptr<Waypoint> waypoint;
  shared_ptr<Queue> queue;
  string name;
  bool finished;

  RemoteDataReader(string name, string address, shared_ptr<Queue> queue)
    : daemon(internode::create_communication_daemon())
    , waypoint(internode::configure_client(daemon, address, UINT_MAX))
    , queue(queue)
    , name(name)
    , finished(false) {
    if (waypoint) {
      waypoint->register_receive_handler(this);
      StartInternalThread();
    }
  }

  ~RemoteDataReader() {
    StopInternalThread();
  }

  void send_req() {
    DataReq req;
    req.set_layer_name(name);
    req.set_iters(1);

    SendCallback callback;
    req.SerializeToString(callback.buffer.get());
    DLOG(INFO) << "sent request for data " << name;
    waypoint->async_send(
      callback.buffer->c_str(), callback.buffer->size(), callback);
  }

  virtual void received(char* buffer, size_t size, Waypoint*) {
    using google::protobuf::io::ArrayInputStream;
    using google::protobuf::io::CodedInputStream;
    DataMsg msg;
    ArrayInputStream zero_stream(buffer, size);
    CodedInputStream coded_stream(&zero_stream);
    coded_stream.SetTotalBytesLimit(256*1024*1024, 256*1024*1024);
    bool result = msg.ParseFromCodedStream(&coded_stream);
    if (!result) {
      LOG(ERROR) << "received blob update failed when parsing, ignoring";
      return;
    }
    DLOG(INFO) << "received data";

    if (queue->add_data(&msg)) {
      send_req();
    } else {
      finished = true;
    }
  }

  virtual void InternalThreadEntry() {
    send_req();
    while (!must_stop() && !finished) {
      poll_one(daemon);
    }
  }
};

template <typename Dtype>
class ReaderKeeper {
  typedef typename RemoteDataLayer<Dtype>::RemoteDataQueue Queue;
  typedef boost::unordered_map<string, shared_ptr<Queue> > QueueMap;

  QueueMap queues;

  ReaderKeeper() {}

 public:
  static ReaderKeeper& instance() {
    static ReaderKeeper instance_;
    return instance_;
  }

  shared_ptr<Queue> get_queue(const LayerParameter& param) {
    RemoteDataParameter rparam = param.remote_data_param();

    typename QueueMap::iterator it = queues.find(rparam.address());
    if (it != queues.end()) {
      return it->second;
    }

    shared_ptr<Queue> data_queue;
    if (rparam.policy() == RemoteDataParameter::PULL_CONTINOUSLY) {
      data_queue.reset(new SimpleDataCache<Dtype>(rparam));
    } else if (rparam.policy() == RemoteDataParameter::USE_CACHE_WHEN_FULL) {
      data_queue.reset(new ReusingShufflingDataCache<Dtype>(rparam));
    } else {
      LOG(ERROR) << "invalid remote data caching policy: " << rparam.policy();
      throw std::runtime_error("invalid remote data caching policy");
    }

    shared_ptr<RemoteDataReader<Dtype> > reader(
      new RemoteDataReader<Dtype>(param.name(), rparam.address(), data_queue));

    return (queues[rparam.address()] =
      shared_ptr<Queue>(reader, data_queue.get()));
  }

  void remove_queue(string address) {
    queues.erase(address);
  }
};

}  // namespace

template <typename Dtype>
RemoteDataLayer<Dtype>::RemoteDataLayer(const LayerParameter& param)
  : BaseDataLayer<Dtype>(param)
  , queue(ReaderKeeper<Dtype>::instance().get_queue(param))
  , transform_blob(new Blob<Dtype>())
  , label_blob(new Blob<Dtype>()) {
  aux_blobs.push_back(transform_blob.get());
  aux_blobs.push_back(label_blob.get());
}

template <typename Dtype>
RemoteDataLayer<Dtype>::~RemoteDataLayer() {
  RemoteDataParameter rparam = Layer<Dtype>::layer_param().remote_data_param();
  ReaderKeeper<Dtype>::instance().remove_queue(rparam.address());
}

template <typename Dtype>
void RemoteDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  CHECK(aux_blobs.size() == top.size());
  aux_blobs[1] = top[1];
  queue->get("Remote data layer prefetch queue empty", aux_blobs);
  this->data_transformer_->Transform(transform_blob.get(), top[0]);
}

template <typename Dtype>
void RemoteDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

#if 0 /* Temporarly removed from list of available layers.
         To be fixed by separate series of patches */
INSTANTIATE_CLASS(RemoteDataLayer);
REGISTER_LAYER_CLASS(RemoteData);
#endif

}  // namespace caffe

