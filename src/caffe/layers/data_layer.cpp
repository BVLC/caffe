#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/dataset_factory.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#ifdef TIMING
#include "caffe/util/benchmark.hpp"
#endif
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::~DataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the dataset resources
  dataset_->close();
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize DB
  dataset_ = DatasetFactory<string, Datum>(
      this->layer_param_.data_param().backend());
  const string& source = this->layer_param_.data_param().source();
  LOG(INFO) << "Opening dataset " << source;
  CHECK(dataset_->open(source, Dataset<string, Datum>::ReadOnly));
  iter_ = dataset_->begin();

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      if (++iter_ == dataset_->end()) {
        iter_ = dataset_->begin();
      }
    }
  }
  // Read a data point, and use it to initialize the top blob.
  CHECK(iter_ != dataset_->end());
  const Datum& datum = iter_->value;

  if (DecodeDatum(datum)) {
    LOG(INFO) << "Decoding Datum";
  }
  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    top[0]->Reshape(this->layer_param_.data_param().batch_size(),
                       datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), crop_size, crop_size);
    this->transformed_data_.Reshape(1, datum.channels(), crop_size, crop_size);
  } else {
    top[0]->Reshape(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), datum.height(), datum.width());
    this->transformed_data_.Reshape(1, datum.channels(),
      datum.height(), datum.width());
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    top[1]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.data_param().batch_size(),
        1, 1, 1);
  }
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DataLayer<Dtype>::InternalThreadEntry() {
  #ifdef TIMING
  Timer batch_timer;
  batch_timer.Start();
  float read_time = 0;
  float trans_time = 0;
  Timer timer;
  #endif
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.data_param().batch_size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a blob
    CHECK(iter_ != dataset_->end());
    const Datum& datum = iter_->value;
    cv::Mat cv_img;
    if (datum.encoded()) {
       cv_img = DecodeDatumToCVMat(datum);
    }
    #ifdef TIMING
    read_time += timer.MilliSeconds();
    timer.Start();
    #endif

    // Apply data transformations (mirror, scale, crop...)
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    if (datum.encoded()) {
      this->data_transformer_.Transform(cv_img, &(this->transformed_data_));
    } else {
      this->data_transformer_.Transform(datum, &(this->transformed_data_));
    }
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    #ifdef TIMING
    trans_time += timer.MilliSeconds();
    #endif
    // go to the next iter
    ++iter_;
    if (iter_ == dataset_->end()) {
      iter_ = dataset_->begin();
    }
  }
  #ifdef TIMING
  LOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << "ms.";
  LOG(INFO) << "Read time: " << read_time << "ms.";
  LOG(INFO) << "Transform time: " << trans_time << "ms.";
  #endif
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(DATA, DataLayer);
}  // namespace caffe
