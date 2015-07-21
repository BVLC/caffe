#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void DocDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  this->JoinPrefetchThread();
  // Reshape to loaded data.
  top[0]->ReshapeLike(this->prefetch_data_);
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  for (int i = 0; i < num_labels_; i++) {
    Blob<Dtype>* prefetch_label = prefetch_labels_[i];
    top[i + 1]->ReshapeLike(*prefetch_label);

    caffe_copy(prefetch_label->count(), prefetch_label->cpu_data(),
               top[i + 1]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(DocDataLayer);

}  // namespace caffe
