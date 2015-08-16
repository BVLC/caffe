#include <opencv2/core/core.hpp>

#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void MatDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) { }

template <typename Dtype>
void MatDataLayer<Dtype>::AddMatVector(const vector<cv::Mat>& mat_vector) {
  // reshape layers
  batch_size_ = mat_vector.size();
  channels_ = mat_vector[0].channels();
  height_ = mat_vector[0].rows;
  width_ = mat_vector[0].cols;
  size_ = channels_ * height_ * width_;
  CHECK_GT(batch_size_ * size_, 0) <<
      "batch_size, channels, height, and width must be positive";
  added_data_.Reshape(batch_size_, channels_, height_, width_);
  // TODO: is this necessary
  added_data_.cpu_data();

  // Apply data transformations (mirror, scale, crop...)
  data_transformer()->Transform(mat_vector, &added_data_);
  Dtype* top_data = added_data_.mutable_cpu_data();
  Reset(top_data);
}

template <typename Dtype>
void MatDataLayer<Dtype>::Reset(Dtype* data) {
  CHECK(data);
  data_ = data;
}

template <typename Dtype>
void MatDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(data_) << "MatDataLayer needs to be initalized by calling Reset";
  top[0]->Reshape(batch_size_, channels_, height_, width_);
  top[0]->set_cpu_data(data_);
}

INSTANTIATE_CLASS(MatDataLayer);
REGISTER_LAYER_CLASS(MatData);

}  // namespace caffe
