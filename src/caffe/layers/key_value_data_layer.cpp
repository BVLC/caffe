#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/key_value_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
KeyValueDataLayer<Dtype>::~KeyValueDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void KeyValueDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Read the file with keys to look up in database
  const string& key_file = this->layer_param_.key_value_data_param().key_file();
  const int column_index = this->layer_param_.key_value_data_param().column();
  CHECK_GE(column_index, 1) << "the column index starts with one";
  LOG(INFO) << "Opening file " << key_file;
  std::ifstream infile(key_file.c_str());
  string line;
  int line_count = 0;
  while (getline(infile, line)) {
    ++line_count;
    for (int i = 0; i < column_index-1; ++i) {
      size_t end = line.find(';');
      CHECK_NE(end, string::npos)
        << "line number " << line_count << " has too few columns";
      line = line.substr(end+1);
    }
    size_t end = line.find(';');
    if (end == string::npos)
      keys_.push_back(line);
    else
      keys_.push_back(line.substr(0, end));
  }
  CHECK_GE(keys_.size(), 1) << "you must specify at least one key to be read";

  const int batch_size = this->layer_param_.data_param().batch_size();

  // Initialize DB
  db_.reset(db::GetDB(this->layer_param_.data_param().backend()));
  db_->Open(this->layer_param_.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());

  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  this->cursor_->Get(keys_[0]);
  datum.ParseFromString(this->cursor_->value());

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

// This function is called on prefetch thread
template <typename Dtype>
void KeyValueDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum datum;
  datum.ParseFromString(this->cursor_->value());
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
  timer.Start();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get the datum at the next key
    Datum datum;
    this->cursor_->Get(keys_[key_index_++]);
    CHECK(this->cursor_->valid())
      << "invalid key " << keys_[key_index_-1] << ";";
    datum.ParseFromString(this->cursor_->value());
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
    timer.Start();
    if (key_index_ == keys_.size()) {
      key_index_ = 0;
      DLOG(INFO) << "Restarting data prefetching from start.";
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(KeyValueDataLayer);
REGISTER_LAYER_CLASS(KeyValueData);

}  // namespace caffe
