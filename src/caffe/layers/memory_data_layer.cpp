#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif // USE_OPENCV

#include <vector>

#include "caffe/layers/memory_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void MemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  channels_ = this->layer_param_.memory_data_param().channels();
  height_ = this->layer_param_.memory_data_param().height();
  width_ = this->layer_param_.memory_data_param().width();
  size_ = channels_ * height_ * width_;
  top[0]->Reshape(1, channels_, height_, width_);
}

#ifdef USE_OPENCV
template <typename Dtype>
std::shared_ptr<Blob<Dtype>>
MemoryDataLayer<Dtype>::FromMatVector(const vector<cv::Mat> &mat_vector) {
  size_t num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no mat to add";
  std::shared_ptr<Blob<Dtype>> data;
  data.reset(new Blob<Dtype>());
  data->Reshape(num, channels_, height_, width_);
  // Apply data transformations (mirror, scale, crop...)
  this->data_transformer_->Transform(mat_vector, data.get());
  return data;
}
#endif // USE_OPENCV

INSTANTIATE_CLASS(MemoryDataLayer);
REGISTER_LAYER_CLASS(MemoryData);

} // namespace caffe
