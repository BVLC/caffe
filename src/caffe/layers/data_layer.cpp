// Copyright 2013 Yangqing Jia

#include <stdint.h>
#include <leveldb/db.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

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
  LOG(INFO) << "Opening leveldb " << this->layer_param_.source();
  leveldb::Status status = leveldb::DB::Open(
      options, this->layer_param_.source(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb "
      << this->layer_param_.source();
  db_.reset(db_temp);
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  iter_->SeekToFirst();
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(iter_->value().ToString());
  // image
  int cropsize = this->layer_param_.cropsize();
  if (cropsize > 0) {
    (*top)[0]->Reshape(
        this->layer_param_.batchsize(), datum.channels(), cropsize, cropsize);
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.batchsize(), datum.channels(), datum.height(),
        datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(this->layer_param_.batchsize(), 1, 1, 1);
  // datum size
  datum_channels_ = datum.channels();
  datum_height_ = datum.height();
  datum_width_ = datum.width();
  datum_size_ = datum.channels() * datum.height() * datum.width();
  CHECK_GT(datum_height_, cropsize);
  CHECK_GT(datum_width_, cropsize);
}

template <typename Dtype>
void DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Datum datum;
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* top_label = (*top)[1]->mutable_cpu_data();
  const Dtype scale = this->layer_param_.scale();
  const Dtype subtraction = this->layer_param_.subtraction();
  int cropsize = this->layer_param_.cropsize();
  for (int itemid = 0; itemid < (*top)[0]->num(); ++itemid) {
    // get a blob
    datum.ParseFromString(iter_->value().ToString());
    const string& data = datum.data();
    if (cropsize) {
      CHECK(data.size()) << "Image cropping only support uint8 data";
      int h_offset = rand() % (datum_height_ - cropsize);
      int w_offset = rand() % (datum_width_ - cropsize);
      for (int c = 0; c < datum_channels_; ++c) {
        for (int h = 0; h < cropsize; ++h) {
          for (int w = 0; w < cropsize; ++w) {
            top_data[((itemid * datum_channels_ + c) * cropsize + h) * cropsize + w] =
                static_cast<Dtype>((uint8_t)data[
                    (c * datum_height_ + h + h_offset) * datum_width_
                    + w + w_offset]
                ) * scale - subtraction;
          }
        }
      }
    } else {
      // we will prefer to use data() first, and then try float_data()
      if (data.size()) {
        for (int j = 0; j < datum_size_; ++j) {
          top_data[itemid * datum_size_ + j] =
              (static_cast<Dtype>((uint8_t)data[j]) * scale) - subtraction;
        }
      } else {
        for (int j = 0; j < datum_size_; ++j) {
          top_data[itemid * datum_size_ + j] =
              (datum.float_data(j) * scale) - subtraction;
        }
      }
    }
    top_label[itemid] = datum.label();
    // go to the next iter
    iter_->Next();
    if (!iter_->Valid()) {
      // We have reached the end. Restart from the first.
      LOG(INFO) << "Restarting data read from start.";
      iter_->SeekToFirst();
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
