

#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/multi_node/async_data.hpp"


namespace caffe {

template <typename Dtype>
AsyncDataLayer<Dtype>::~AsyncDataLayer<Dtype>() {

}


template <typename Dtype>
void AsyncDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(full_->peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
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


template<typename Dtype>
void AsyncDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  
  if (Caffe::root_solver()) {
    LOG(WARNING) << "skip reading data in AsyncData layer with root solvers";
    return;
  }

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(full_->pop("Waiting for data"));
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  
  // process the first datum
  this->transformed_data_.set_cpu_data(top_data);
  this->data_transformer_->Transform(datum, &(this->transformed_data_));
  // Copy label.
  if (this->output_labels_) {
    top_label[0] = datum.label();
  }
  free_->push(const_cast<Datum*>(&datum));

  for (int item_id = 1; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(full_->pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));
    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();

    free_->push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  //LOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  //LOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  //LOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

}

INSTANTIATE_CLASS(AsyncDataLayer);
REGISTER_LAYER_CLASS(AsyncData);

}  // namespace caffe


