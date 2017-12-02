#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void BasePrefetchingDataLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {

#ifdef USE_OPENCL
  // Direct async to GPU currently unsupported on OpenCL
  if (this->device_->backend() == BACKEND_OPENCL) {
    this->Forward_cpu(bottom, top);
    return;
  }
#endif  // USE_OPENCL

  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_gpu_data(prefetch_current_->label_.mutable_gpu_data());
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer,
                              (half_fp), (half_fp), (half_fp));
INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer,
                              (float), (float), (float));
INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer,
                              (double), (double), (double));

}  // namespace caffe
