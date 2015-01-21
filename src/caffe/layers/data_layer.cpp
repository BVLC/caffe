#include <opencv2/core/core.hpp>

#include <stdint.h>
#include <sys/stat.h>

#include <string>
#include <vector>
#include <map>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/dataset_factory.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

map<string, weak_ptr<DataLoader::Body> > DataLoader::instances_;
boost::mutex DataLoader::instances_mutex_;

DataLoader::DataLoader(const DataParameter& param, int index):
    source_(param.source(index)) {
  // Makes sure create only one body per source
  boost::mutex::scoped_lock lock(instances_mutex_);
  weak_ptr<Body> body = instances_[source_];
  body_ = body.lock();
  if (!body_) {
    body_.reset(new Body(param, index));
    instances_[source_] = weak_ptr<Body>(body_);
  }
}

DataLoader::~DataLoader() {
  boost::mutex::scoped_lock lock(instances_mutex_);
  body_.reset();
  if (instances_[source_].expired())
    instances_.erase(source_);
}

DataLoader::Body::Body(const DataParameter& param, int index) {
  // Initialize DB
  DataParameter_DB backend = param.backend_size() ?
      param.backend(index) : DataParameter::LEVELDB;
  dataset_ = DatasetFactory<string, Datum>(backend);
  LOG(INFO) << "Opening dataset " << param.source(index);
  CHECK(dataset_->open(param.source(index), Dataset<string, Datum>::ReadOnly));
  iter_ = dataset_->begin();

  // Check if we need to randomly skip a few data points
  if (param.rand_skip()) {
    unsigned int skip = caffe_rng_rand() % param.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      if (++iter_ == dataset_->end()) {
        iter_ = dataset_->begin();
      }
    }
  }

  // Add prefetch datums to layer free queue
  int prefetch = param.prefetch() * param.batch_size();
  for(int i = 0; i < prefetch; ++i) {
    free_.push(new Datum());
  }

  CHECK(StartInternalThread()) << "DataLoader thread start failed";
}

DataLoader::Body::~Body() {
  CHECK(StopInternalThread()) << "DataLoader thread stop failed";
  Datum* datum;
  while(free_.try_pop(datum)) {
    delete datum;
  }
  while(full_.try_pop(datum)) {
    delete datum;
  }
  // clean up the dataset resources
  dataset_->close();
}

void DataLoader::Body::InternalThreadEntry() {
  try {
    while(!must_stop()) {
      CHECK(iter_ != dataset_->end());

      Datum* datum = free_.pop();
      // TODO deserialize in-place instead of copy?
      datum->CopyFrom(iter_->value);
      full_.push(datum);

      ++iter_;
      if (iter_ == dataset_->end()) {
        iter_ = dataset_->begin();
      }
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

static unsigned int get_datalayer_specific_random_seed() {
  unsigned int seed = Caffe::get_random_seed();
  if(!seed) {
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
    for(int i = 0; i < data.probability().size(); ++i) {
      sum += data.probability(i);
    }
    CHECK(fabsf(sum - 1.0f) < 1e-6f)
      << "Invalid DataParameter, probabilities do not sum to 1";
  }
  for(int i = 0; i < data.source().size(); ++i) {
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

  if (DecodeDatum(datum)) {
    LOG(INFO) << "Decoding Datum";
  }
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.data_param().batch_size();
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, datum->channels(), crop_size, crop_size);
    for(int i = 0; i < this->PREFETCH_COUNT; ++i) {
	    this->prefetch_[i].data_.Reshape(batch_size, datum->channels(),
	        crop_size, crop_size);
	  }
    this->transformed_data_.Reshape(1, datum->channels(),
        crop_size, crop_size);
  } else {
    top[0]->Reshape(batch_size, datum->channels(),
        datum->height(), datum->width());
    for(int i = 0; i < this->PREFETCH_COUNT; ++i) {
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
    top[1]->Reshape(batch_size, 1, 1, 1);
    for(int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(batch_size, 1, 1, 1);
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
  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  const int batch_size = this->layer_param_.data_param().batch_size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    DataLoader* loader = next_loader();
    const Datum& datum = *(loader->full().pop("Waiting on data loader"));

    cv::Mat cv_img;
    if (datum.encoded()) {
       cv_img = DecodeDatumToCVMat(datum);
    }
    read_time += timer.MicroSeconds();
    timer.Start();

    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    if (datum.encoded()) {
      this->data_transformer_.Transform(cv_img, &(this->transformed_data_));
    } else {
      this->data_transformer_.Transform(datum, &(this->transformed_data_));
    }
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();

    loader->free().push((Datum*) &datum);
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
    for(int i = 0; i < loaders_.size(); ++i) {
      DataLoader* loader = loaders_[i].get();
      if(!loader->full().empty()) {
        return loader;
      }
    }
  } else {
    // Pick loader randomly with probability
    float rand = rand_(rand_engine_);
    for(int i = 0; i < data.probability().size(); ++i) {
      rand -= data.probability(i);
      if(rand < 0) {
        return loaders_[i].get();
      }
    }
  }
  // If no data ready, or rounding error on probabilities
  return loaders_[0].get();
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(DATA, DataLayer);
}  // namespace caffe
