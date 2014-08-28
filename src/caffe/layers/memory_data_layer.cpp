#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void MemoryDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
     vector<Blob<Dtype>*>* top) {
  batch_size_ = this->layer_param_.memory_data_param().batch_size();
  this->datum_channels_ = this->layer_param_.memory_data_param().channels();
  this->datum_height_ = this->layer_param_.memory_data_param().height();
  this->datum_width_ = this->layer_param_.memory_data_param().width();
  this->datum_size_ = this->datum_channels_ * this->datum_height_ *
      this->datum_width_;
  CHECK_GT(batch_size_ * this->datum_size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  (*top)[0]->Reshape(batch_size_, this->datum_channels_, this->datum_height_,
                     this->datum_width_);
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
  (*top)[0]->set_cpu_data(data_ + pos_ * this->datum_size_);
  (*top)[1]->set_cpu_data(labels_ + pos_);
  pos_ = (pos_ + batch_size_) % n_;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::AddImagesAndLabels(
    const vector<cv::Mat>& images, const vector<int>& labels) {
  size_t num_images = images.size();
  CHECK_GT(num_images, 0) << "There is no image to add";
  CHECK_LE(num_images, batch_size_)<<
      "The number of added images " << images.size() <<
      " must be no greater than the batch size " << batch_size;
  CHECK_LE(num_images, labels.size()) <<
      "The number of images " << images.size() <<
      " must be no greater than the number of labels " << labels.size();

  Datum datum;
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  // Apply data transformations (mirror, scale, crop...)
  this->data_transformer_.Transform(item_id, datum, mean, top_data);
  int image_id;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    image_id = item_id % num_images;
    OpenCVImageToDatum(images[image_id], labels[image_id], new_height,
                       new_width, &datum);
    this->data_transformer_.Transform(item_id, datum, mean, top_data);
  }
}

INSTANTIATE_CLASS(MemoryDataLayer);

}  // namespace caffe
