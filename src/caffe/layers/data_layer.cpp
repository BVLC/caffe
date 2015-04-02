#include <boost/thread.hpp>
#include <opencv2/core/core.hpp>

#include <stdint.h>
#include <sys/stat.h>

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

map<string, weak_ptr<DataLoader::Body> > DataLoader::instances_;
static boost::mutex data_loader_instances_mutex_;

DataLoader::DataLoader(const DataParameter& param, int index):
    source_(param.source(index)) {
  // Makes sure create only one body per source
  boost::mutex::scoped_lock lock(data_loader_instances_mutex_);
  weak_ptr<Body> body = instances_[source_];
  body_ = body.lock();
  if (!body_) {
    body_.reset(new Body(param, index));
    instances_[source_] = weak_ptr<Body>(body_);
  }
}

DataLoader::~DataLoader() {
  boost::mutex::scoped_lock lock(data_loader_instances_mutex_);
  body_.reset();
  if (instances_[source_].expired())
    instances_.erase(source_);
}

DataLoader::Body::Body(const DataParameter& param, int index) {
  // Initialize DB
  DataParameter_DB backend = param.backend_size() ?
      param.backend(index) : DataParameter::LEVELDB;
  db_.reset(db::GetDB(backend));
  db_->Open(param.source(index), db::READ);
  cursor_.reset(db_->NewCursor());

  // Check if we should randomly skip a few data points
  if (param.rand_skip()) {
    unsigned int skip = caffe_rng_rand() % param.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      cursor_->Next();
    }
  }

  // Add prefetch datums to layer free queue
  int prefetch = param.prefetch() * param.batch_size();
  for (int i = 0; i < prefetch; ++i) {
    free_.push(new Datum());
  }

  CHECK(StartInternalThread()) << "DataLoader thread start failed";
}

DataLoader::Body::~Body() {
  CHECK(StopInternalThread()) << "DataLoader thread stop failed";
  Datum* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

void DataLoader::Body::InternalThreadEntry() {
  try {
    while (!must_stop()) {
      Datum* datum = free_.pop();
      // TODO deserialize in-place instead of copy?
      datum->ParseFromString(cursor_->value());
      full_.push(datum);

      // go to the next iter
      cursor_->Next();
      if (!cursor_->valid()) {
        DLOG(INFO) << "Restarting data prefetching from start.";
        cursor_->SeekToFirst();
      }
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

static unsigned int get_datalayer_specific_random_seed() {
  unsigned int seed = Caffe::get_random_seed();
  if (!seed) {
    seed = caffe_rng_rand();
  }
  return seed + 87267527;
}

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    rand_engine_(get_datalayer_specific_random_seed()) {
  const DataParameter& data = param.data_param();
  if (data.backend_size()) {
    CHECK(data.source().size() == data.backend().size())
      << "Invalid DataParameter, there should be one backend per source";
  }
  if (data.probability_size()) {
    CHECK(data.source().size() == data.backend().size())
      << "Invalid DataParameter, there should be one probability per source";
    float sum = 0;
    for (int i = 0; i < data.probability().size(); ++i) {
      sum += data.probability(i);
    }
    CHECK(fabsf(sum - 1.0f) < 1e-6f)
      << "Invalid DataParameter, probabilities do not sum to 1";
  }
  for (int i = 0; i < data.source().size(); ++i) {
    DataLoader* ld = new DataLoader(data, i);
    loaders_.push_back(shared_ptr<DataLoader>(ld));
  }
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  CHECK(this->StopInternalThread()) << "Stop thread failed";
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Look at first data point to initialize the top blob.
  Datum* datum = loaders_[0].get()->full().peek();

  bool force_color = this->layer_param_.data_param().force_encoded_color();
  if ((force_color && DecodeDatum(datum, true)) ||
      DecodeDatumNative(datum)) {
    LOG(INFO) << "Decoding Datum";
  }
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.data_param().batch_size();
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, datum->channels(), crop_size, crop_size);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, datum->channels(),
          crop_size, crop_size);
    }
    this->transformed_data_.Reshape(1, datum->channels(),
        crop_size, crop_size);
  } else {
    top[0]->Reshape(batch_size, datum->channels(),
        datum->height(), datum->width());
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, datum->channels(),
          datum->height(), datum->width());
    }
    this->transformed_data_.Reshape(1, datum->channels(),
        datum->height(), datum->width());
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template <typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  const int batch_size = this->layer_param_.data_param().batch_size();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  bool force_color = this->layer_param_.data_param().force_encoded_color();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    DataLoader* loader = next_loader();
    const Datum& datum = *(loader->full().pop("Waiting on data loader"));

    // Reshape on single input batches for inputs of varying dimension.
    if (batch_size == 1 && crop_size == 0) {
      batch->data_.Reshape(1, datum.channels(),
          datum.height(), datum.width());
      this->transformed_data_.Reshape(1, datum.channels(),
          datum.height(), datum.width());
    }

    cv::Mat cv_img;
    if (datum.encoded()) {
      if (force_color) {
        cv_img = DecodeDatumToCVMat(datum, true);
      } else {
        cv_img = DecodeDatumToCVMatNative(datum);
      }
      if (cv_img.channels() != this->transformed_data_.channels()) {
        LOG(WARNING) << "Your dataset contains encoded images with mixed "
        << "channel sizes. Consider adding a 'force_color' flag to the "
        << "model definition, or rebuild your dataset using "
        << "convert_imageset.";
      }
    }
    read_time += timer.MicroSeconds();
    timer.Start();

    // Apply data transformations (mirror, scale, crop...)
    Dtype* top_data = batch->data_.mutable_cpu_data();
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    if (datum.encoded()) {
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    } else {
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
    }
    if (this->output_labels_) {
      batch->label_.mutable_cpu_data()[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();

    loader->free().push(const_cast<Datum*>(&datum));
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

// This function is called on prefetch thread
template <typename Dtype>
DataLoader* DataLayer<Dtype>::next_loader() {
  const DataParameter& data = this->layer_param().data_param();
  // Default case without probabilities, try to find a loader with
  // data ready, or return first one
  if (data.probability_size() == 0) {
    for (int i = 0; i < loaders_.size(); ++i) {
      DataLoader* loader = loaders_[i].get();
      if (!loader->full().empty()) {
        return loader;
      }
    }
  } else {
    // Pick loader randomly with probability
    float rand = rand_(rand_engine_);
    for (int i = 0; i < data.probability().size(); ++i) {
      rand -= data.probability(i);
      if (rand < 0) {
        return loaders_[i].get();
      }
    }
  }
  // If no data ready, or rounding error on probabilities
  return loaders_[0].get();
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
