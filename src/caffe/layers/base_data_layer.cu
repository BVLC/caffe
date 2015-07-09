#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template<typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();

  if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // Reshape to loaded data.
    top[0]->ReshapeLike(this->prefetch_data_);
    // Copy the data
    caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
               top[0]->mutable_gpu_data());
    if (this->output_labels_) {
      // Reshape to loaded labels.
      top[1]->ReshapeLike(prefetch_label_);
      // Copy the labels.
      caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
                 top[1]->mutable_gpu_data());
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_context_->id());

    // Reshape to loaded data.
    top[0]->ReshapeLike(this->prefetch_data_);
    // Copy the data
    greentea_copy<Dtype>(prefetch_data_.count(),
                         prefetch_data_.cpu_data(),
                         (cl_mem) (top[0]->mutable_gpu_data()), 0, &ctx);
    if (this->output_labels_) {
      // Reshape to loaded labels.
      top[1]->ReshapeLike(prefetch_label_);
      // Copy the labels.
      greentea_copy<Dtype>(prefetch_label_.count(),
                           prefetch_label_.cpu_data(),
                           (cl_mem) (top[1]->mutable_gpu_data()), 0, &ctx);
    }
#endif  // USE_GREENTEA
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
