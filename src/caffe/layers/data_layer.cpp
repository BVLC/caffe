// Copyright 2013 Yangqing Jia

#include <vector>

#include <leveldb/db.h>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Neuron Layer takes no input blobs.";
  CHECK_EQ(top->size(), 2) << "Neuron Layer takes two blobs as output.";
  // Initialize the leveldb
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.create_if_missing = false;
  leveldb::Status status = leveldb::DB::Open(
      options, this->layer_param_.source(), &db_temp);
  CHECK(status.ok());
  db_.reset(db_temp);
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(iter_->value().ToString());
  const BlobProto& blob = datum.blob();
  // image
  (*top)[0]->Reshape(
      this->layer_param_.batchsize(), blob.channels(), blob.height(),
      blob.width());
  // label
  (*top)[1]->Reshape(this->layer_param_.batchsize(), 1, 1, 1);
  // datum size
  datum_size_ = blob.data_size();
}

template <typename Dtype>
void DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Datum datum;
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* top_label = (*top)[0]->mutable_cpu_diff();
  for (int i = 0; i < this->layer_param_.batchsize(); ++i) {
    // get a blob
    datum.ParseFromString(iter_->value().ToString());
    const BlobProto& blob = datum.blob();
    for (int j = 0; j < datum_size_; ++j) {
      top_data[i * datum_size_ + j] = blob.data(j);
    }
    top_label[i] = datum.label();
    // go to the next iter
    iter_->Next();
    if (!iter_->Valid()) {
      // We have reached the end. Restart from the first.
      LOG(INFO) << "Restarting data read from start.";
      iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    }
  }
}

template <typename Dtype>
void DataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Forward_cpu(bottom, top);
  // explicitly copy data to gpu - this is achieved by simply calling gpu_data
  // functions.
  // TODO(Yangqing): maybe we don't need this since data synchronization is
  // simply done under the hood?
  (*top)[0]->gpu_data();
  (*top)[1]->gpu_data();
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype DataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

template <typename Dtype>
Dtype DataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(DataLayer);

}  // namespace caffe
