#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <stdint.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "boost/make_shared.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/unordered_map.hpp"
#include "caffe/internode/configuration.hpp"
#include "caffe/layers/remote_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"

namespace caffe {

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
    queue->get("Remote data layer prefetch queue empty", top);
  }

  CHECK(bottom.empty());
}

using internode::RemoteId;
using internode::configure_client;
using internode::Waypoint;
using internode::Daemon;
using internode::create_communication_daemon;
using internode::is_remote_address;

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
  int gen;  // randomness generator - large prime number
  int curr;
  bool read_all;
  boost::mutex mtx;

  template <typename ShapeInfo>
  void create_cache(ShapeInfo info) {
    vector<vector<int> > shapes(info.shape_size());
    batch.top.resize(shapes.size());
    for (int j = 0; j < shapes.size(); ++j) {
      batch.top[j] = new Blob<Dtype>();
      batch.top[j]->Reshape(info.shape(j));
      shapes[j] = batch.top[j]->shape();

      if (batch_size != 0) {
        CHECK(shapes[j][0] == batch_size);
      } else {
        batch_size = shapes[j][0];
        cache_size *= batch_size;
      }
      shapes[j][0] = 1;
    }

    for (int i = 0; i < cache_size; ++i) {
      Data<Dtype> data;
      data.top.resize(shapes.size());

      for (int j = 0; j < data.top.size(); ++j) {
        data.top[j] = new Blob<Dtype>();
        data.top[j]->Reshape(shapes[j]);
      }
      cache.push_back(data);
    }
  }

  explicit ReusingShufflingDataCache(RemoteDataParameter param)
    : read(0)
    , batch_size(0)
    , cache_size(param.cache_size())
    , gen(10000019)
    , curr(1)
    , read_all(false) {
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
    int curr_read = 0;
    while (true) {
      boost::mutex::scoped_lock lock(mtx);
      curr_read = read;
      if (read_all) {
        break;
      }
    }

    CHECK_GT(batch_size, 0);
    for (int j = 0; j < ret.size(); ++j) {
      ret[j]->ReshapeLike(*batch.top[j]);
    }
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < ret.size(); ++j) {
        Blob<Dtype>* target = ret[j];
        Blob<Dtype>* src = cache[curr].top[j];

        int offset = batch.top[j]->offset(i);
        caffe_copy(src->count(),
                   src->cpu_data(),
                   target->mutable_cpu_data() + offset);
      }

      curr = (curr + gen) % curr_read;
    }
  }
};

template <typename Dtype>
struct SimpleDataCache : RemoteDataLayer<Dtype>::RemoteDataQueue {
  BlockingQueue<Element*> cached;
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
  }

  explicit SimpleDataCache(RemoteDataParameter param)
    : cache_size(param.cache_size()) {

    if (param.shape_size() > 0) {
      create_cache(param);
    }
  }

  virtual bool add_data(DataMsg* update) {
    if (cached.size() + free.size() == 0) {
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
    Data<Dtype>* data = cached.pop(msg)->template cast<Data<Dtype> >();
    CHECK(data->top.size() == ret.size());
    for (int i = 0; i < ret.size(); ++i) {
      ret[i]->ReshapeLike(*data->top[i]);
      caffe_copy(data->top[i]->count(), data->top[i]->cpu_data(),
                 ret[i]->mutable_cpu_data());
    }

    free.push(data);
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
    : daemon(create_communication_daemon())
    , waypoint(is_remote_address(address) ?
        configure_client(daemon, address) : boost::shared_ptr<Waypoint>())
    , queue(queue)
    , name(name)
    , finished(false) {
    if (waypoint) {
      waypoint->register_receive_handler(this);
      StartInternalThread();
    }
  }


  void send_req() {
    DataReq req;
    req.set_layer_name(name);
    req.set_iters(1);
    string str;
    req.SerializeToString(&str);
    VLOG(2) << "sent request for data " << name;
    waypoint->send(str.c_str(), str.size());
  }

  virtual void received(char* buffer, size_t size, RemoteId) {
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
    VLOG(2) << "received data";

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
};

}  // namespace

template <typename Dtype>
RemoteDataLayer<Dtype>::RemoteDataLayer(const LayerParameter& param)
  : BaseDataLayer<Dtype>(param)
  , queue(ReaderKeeper<Dtype>::instance()
      .get_queue(param))
  , transform_blob(new Blob<Dtype>()) {
}

template <typename Dtype>
void RemoteDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  transform_blob->ReshapeLike(*top[0]);
  queue->get("Remote data layer prefetch queue empty", top);
  caffe_copy(transform_blob->count(),
             top[0]->cpu_data(),
             transform_blob->mutable_cpu_data());
  this->data_transformer_->Transform(transform_blob, top[0]);
}

template <typename Dtype>
void RemoteDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

INSTANTIATE_CLASS(RemoteDataLayer);
REGISTER_LAYER_CLASS(RemoteData);

}  // namespace caffe

