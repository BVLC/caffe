#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MemoryDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
     vector<Blob<Dtype>*>* top) {
  batch_size_ = this->layer_param_.memory_data_param().batch_size();
  datum_channels_ = this->layer_param_.memory_data_param().channels();
  datum_height_ = this->layer_param_.memory_data_param().height();
  datum_width_ = this->layer_param_.memory_data_param().width();
  datum_size_ = datum_channels_ * datum_height_ * datum_width_;
  CHECK_GT(batch_size_ * datum_size_, 0) << "batch_size, channels, height,"
    " and width must be specified and positive in memory_data_param";
  (*top)[0]->Reshape(batch_size_, datum_channels_, datum_height_, datum_width_);
  (*top)[1]->Reshape(batch_size_, 1, 1, 1);
  data_ = NULL;
  labels_ = NULL;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
  CHECK(data);
  CHECK(labels);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  data_ = data;
  labels_ = labels;
  n_ = n;
  pos_ = 0;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK(data_) << "MemoryDataLayer needs to be initalized by calling Reset";
  (*top)[0]->set_cpu_data(data_ + pos_ * datum_size_);
  (*top)[1]->set_cpu_data(labels_ + pos_);
  pos_ = (pos_ + batch_size_) % n_;
}

INSTANTIATE_CLASS(MemoryDataLayer);

}  // namespace caffe
