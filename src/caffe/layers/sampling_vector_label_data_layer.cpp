#include <leveldb/db.h>
#include <stdint.h>
#include <cmath>

#include <string>
#include <vector>
#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
SamplingVectorLabelDataLayer<Dtype>::~SamplingVectorLabelDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void SamplingVectorLabelDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Initialize DB
  leveldb::DB* db_temp;
  leveldb::Options options = GetLevelDBOptions();
  options.create_if_missing = false;
  LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
  leveldb::Status status = leveldb::DB::Open(
      options, this->layer_param_.data_param().source(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb "
                     << this->layer_param_.data_param().source() << std::endl
                     << status.ToString();
  db_.reset(db_temp);
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  iter_->SeekToFirst();

  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(iter_->value().ToString());

  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(this->layer_param_.data_param().batch_size(),
                       datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), crop_size, crop_size);
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), datum.height(), datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (this->output_labels_) {
    int label_size = std::max(datum.multi_label_size(), datum.multi_float_label_size());
    CHECK_GE(label_size, 1) << "Vector label size must greater than 0.";
    (*top)[1]->Reshape(this->layer_param_.data_param().batch_size(), label_size, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.data_param().batch_size(),
        label_size, 1, 1);
  }
  if (top->size() == 3) {
    (*top)[2]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
  }
  if (this->layer_param_.sampling_param().has_min_index()) {
    minIdx_ = this->layer_param_.sampling_param().min_index();
  } else {
    minIdx_ = 0;
  }
  if (this->layer_param_.sampling_param().has_max_index()) {
    maxIdx_ = this->layer_param_.sampling_param().max_index();
  } else {
    const int label_size = std::max(datum.multi_label_size(), datum.multi_float_label_size());
    maxIdx_ = label_size - 1;
  }
  curIdx_ = minIdx_;

  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
  
  // setup sampling pools
  std::ifstream pool_file(this->layer_param_.sampling_param().pool_file().c_str());
  CHECK(pool_file.good()) << "Failed to open pool file "
      << this->layer_param_.sampling_param().pool_file() << std::endl;
  int idx = 0, pool_size, label_id;
  while (pool_file >> label_id >> pool_size) {
    string key;
    for (int i = 0; i < pool_size; ++i) {
      pool_file >> key;
      sampling_pool[idx].push_back(key);
    }
    idx++;
  }
}

template <typename Dtype>
void SamplingVectorLabelDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  this->JoinPrefetchThread();
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             (*top)[0]->mutable_cpu_data());
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
               (*top)[1]->mutable_cpu_data());
  }
  if (top->size() == 3) {
    const int batch_size = this->layer_param_.data_param().batch_size();
    Dtype *cur_labels = (*top)[2]->mutable_cpu_data();
    for (int i = 0; i < batch_size; ++i) {
      cur_labels[i] = curIdx_;
    }
  }
  curIdx_++;
  if (curIdx_ > maxIdx_) {
    curIdx_ = minIdx_;
  }
  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

template <typename Dtype>
unsigned int SamplingVectorLabelDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void SamplingVectorLabelDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.data_param().batch_size();
  vector<string> cur_pool = sampling_pool[curIdx_];
  const int pool_size = cur_pool.size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    const unsigned int rand_index = PrefetchRand();
    string key = cur_pool[rand_index % pool_size];
    string value;
    leveldb::Status s = db_->Get(leveldb::ReadOptions(), key, &value);
    CHECK(s.ok()) << "Failed to find value for key: " << key;
    datum.ParseFromString(value);
    
    int label_size = std::max(datum.multi_label_size(), datum.multi_float_label_size());
    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);

    if (this->output_labels_) {
      if (datum.multi_label_size()) {
        for (int idx = 0; idx < label_size; ++idx) {
          int top_label_idx = item_id * label_size + idx;
          top_label[top_label_idx] = datum.multi_label(idx);
        }
      } else {
        for (int idx = 0; idx < label_size; ++idx) {
          int top_label_idx = item_id * label_size + idx;
          top_label[top_label_idx] = datum.multi_float_label(idx);
        }
      }
    }
  }
}

INSTANTIATE_CLASS(SamplingVectorLabelDataLayer);

}  // namespace caffe
