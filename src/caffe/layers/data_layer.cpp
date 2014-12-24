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

DataLoader::DataLoader(const DataParameter& param, int index,
                       blocking_queue<Datum*>* free,
                       blocking_queue<Datum*>* full):
    source_(param.source(index)) {
  boost::mutex::scoped_lock lock(instances_mutex_);
  weak_ptr<Body> body = instances_[source_];
  body_ = body.lock();
  if (body_) {
    CHECK(!free || free == body_.get()->free_);
    CHECK(!full || full == body_.get()->full_);
  } else {
    body_.reset(new Body(param, index, free, full));
    instances_[source_] = weak_ptr<Body>(body_);
  }
}

DataLoader::~DataLoader() {
  boost::mutex::scoped_lock lock(instances_mutex_);
  body_.reset();
  if (instances_[source_].expired())
    instances_.erase(source_);
}

DataLoader::Body::Body(const DataParameter& param, int index,
                       blocking_queue<Datum*>* free,
                       blocking_queue<Datum*>* full) :
    free_(free),
    full_(full),
    own_free_full_() {

  // Initialize queues
  if(!free_) {
    free_ = new blocking_queue<Datum*>();
    full_ = new blocking_queue<Datum*>();
    own_free_full_ = true;
  }

  // Initialize DB
  dataset_ = DatasetFactory<string, Datum>(param.backend());
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
    free_->push(new Datum());
  }

  CHECK(StartInternalThread()) << "DataLoader thread start failed";
}

DataLoader::Body::~Body() {
  CHECK(StopInternalThread()) << "DataLoader thread stop failed";
  Datum* datum;
  while(free_->try_pop(datum)) {
    delete datum;
  }
  while(full_->try_pop(datum)) {
    delete datum;
  }

  // clean up the dataset resources
  dataset_->close();

  if(own_free_full_) {
    delete free_;
    delete full_;
  }
}

void DataLoader::Body::InternalThreadEntry() {
  while(!must_stop()) {
    CHECK(iter_ != dataset_->end());

    Datum* datum = free_->pop();
    // TODO deserialize in-place instead of copy?
    datum->CopyFrom(iter_->value);
    full_->push(datum);

    ++iter_;
    if (iter_ == dataset_->end()) {
      iter_ = dataset_->begin();
    }
  }
}

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param) {
  DataLoader* ld = new DataLoader(param.data_param(), 0);
  loaders_.push_back(shared_ptr<DataLoader>(ld));
  loaders_free_ = ld->free();
  loaders_full_ = ld->full();

  // Loaders share queues in case of multiple sources
  for(int i = 1; i < param.data_param().source().size(); ++i) {
    ld = new DataLoader(param.data_param(), i, loaders_free_, loaders_full_);
    loaders_.push_back(shared_ptr<DataLoader>(ld));
  }
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // Look at first data point to initialize the top blob.
  Datum* datum = loaders_full_->peek();

  if (DecodeDatum(datum))
    LOG(INFO) << "Decoding Datum";

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
  if (this->output_labels_)
    top_label = batch->label_.mutable_cpu_data();

  const int batch_size = this->layer_param_.data_param().batch_size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    const Datum& datum = *(loaders_full_->pop("Waiting on data loader"));

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

    loaders_free_->push((Datum*) &datum);
  }
  batch_timer.Stop();
//  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
//  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
//  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(DATA, DataLayer);
}  // namespace caffe
