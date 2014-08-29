#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void MemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
  added_data_.Reshape(batch_size_, this->datum_channels_, this->datum_height_,
                      this->datum_width_);
  added_label_.Reshape(batch_size_, 1, 1, 1);
  data_ = NULL;
  labels_ = NULL;
  added_data_.cpu_data();
  added_label_.cpu_data();
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
  has_new_data_ = false;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::AddImages(
    const vector<cv::Mat>& images, const bool is_color_images) {
  const vector<int> fake_labels(images.size(), 0);
  AddImagesAndLabels(images, fake_labels, is_color_images);
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::AddImagesAndLabels(
    const vector<cv::Mat>& images, const vector<int>& labels,
    const bool is_color_images) {
  CHECK(!has_new_data_) <<
      "There are pending images that haven't been consumed";
  size_t num_images = images.size();
  CHECK_GT(num_images, 0) << "There is no image to add";
  CHECK_EQ(num_images, batch_size_) <<
      "The number of added images must be equal to the batch size";
  CHECK_EQ(num_images, labels.size()) <<
      "The number of images must be equal to the number of labels";

  Datum datum;
  Dtype* top_data = added_data_.mutable_cpu_data();
  Dtype* top_label = added_label_.mutable_cpu_data();
  for (int batch_item_id = 0; batch_item_id < num_images; ++batch_item_id) {
    OpenCVImageToDatum(
        images[batch_item_id], labels[batch_item_id], this->datum_height_,
        this->datum_width_, &datum);
    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(batch_item_id, datum, this->mean_,
                                      top_data);
    top_label[batch_item_id] = labels[batch_item_id];
  }
  // num_images == batch_size_
  Reset(top_data, top_label, batch_size_);
  has_new_data_ = true;
}

INSTANTIATE_CLASS(MemoryDataLayer);

}  // namespace caffe
