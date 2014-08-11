#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
Dtype ImageDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  if (this->layer_param_.image_data_param().has_source()) {
    JoinPrefetchThread();
  }
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      (*top)[0]->mutable_gpu_data());
  caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
      (*top)[1]->mutable_gpu_data());
  if (this->layer_param_.image_data_param().has_source()) {
    // Start a new prefetch thread
    CreatePrefetchThread();
  }
  return Dtype(0.);
}

INSTANTIATE_CLASS(ImageDataLayer);

}  // namespace caffe
