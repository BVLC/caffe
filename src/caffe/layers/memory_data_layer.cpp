#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif // USE_OPENCV

#include <vector>

#include "caffe/layers/memory_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void MemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  batch_size_ = this->layer_param_.memory_data_param().batch_size();
  channels_ = this->layer_param_.memory_data_param().channels();
  height_ = this->layer_param_.memory_data_param().height();
  width_ = this->layer_param_.memory_data_param().width();
  size_ = channels_ * height_ * width_;
  CHECK_GT(batch_size_ * size_, 0)
      << "batch_size, channels, height, and width must be specified and"
         " positive in memory_data_param";
  top[0]->Reshape(batch_size_, channels_, height_, width_);
  added_data_.Reshape(batch_size_, channels_, height_, width_);
  data_ = NULL;
  added_data_.cpu_data();
}

#ifdef USE_OPENCV
template <typename Dtype>
void MemoryDataLayer<Dtype>::AddMatVector(const vector<cv::Mat> &mat_vector) {
  size_t num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no mat to add";
  CHECK_EQ(num % batch_size_, 0)
      << "The added data must be a multiple of the batch size.";
  added_data_.Reshape(num, channels_, height_, width_);
  // Apply data transformations (mirror, scale, crop...)
  this->data_transformer_->Transform(mat_vector, &added_data_);
  Dtype *top_data = added_data_.mutable_cpu_data();
  Reset(top_data, num);
}
#endif // USE_OPENCV

template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(Dtype *data, int n) {
  CHECK(data);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  // Warn with transformation parameters since a memory array is meant to
  // be generic and no transformations are done with Reset().
  if (this->layer_param_.has_transform_param()) {
    LOG(WARNING) << this->type() << " does not transform array data on Reset()";
  }
  data_ = data;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::set_batch_size(int new_size) {
  batch_size_ = new_size;
  added_data_.Reshape(batch_size_, channels_, height_, width_);
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                         const vector<Blob<Dtype> *> &top) {
  Forward_const_cpu(bottom, top);
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_const_cpu(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) const {
  CHECK(data_) << "MemoryDataLayer needs to be initialized by calling Reset";
  top[0]->Reshape(batch_size_, channels_, height_, width_);
  top[0]->set_cpu_data(data_);
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                         const vector<Blob<Dtype> *> &top) {
  Forward_const_gpu(bottom, top);
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_const_gpu(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) const {
  Forward_const_cpu(bottom, top);
}
INSTANTIATE_CLASS(MemoryDataLayer);
REGISTER_LAYER_CLASS(MemoryData);

} // namespace caffe
