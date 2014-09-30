#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
VectorLabelDataLayer<Dtype>::~VectorLabelDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void VectorLabelDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
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

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
    }
  }
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
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void VectorLabelDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK(iter_);
    CHECK(iter_->Valid());
    datum.ParseFromString(iter_->value().ToString());
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

    // go to the next iter
    iter_->Next();
    if (!iter_->Valid()) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      iter_->SeekToFirst();
    }
  }
}

INSTANTIATE_CLASS(VectorLabelDataLayer);

}  // namespace caffe
