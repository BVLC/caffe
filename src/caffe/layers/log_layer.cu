#include <vector>

#include "caffe/layers/log_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void LogLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                    const vector<Blob<MItype>*>& bottom,
                                    const vector<Blob<MOtype>*>& top) {
  const int_tp count = bottom[0]->count();
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();

  if (input_scale_ == Dtype(1) && input_shift_ == Dtype(0)) {
    this->device_->log(count, bottom_data, top_data);
  } else {
    this->device_->template copy<Dtype>(count, bottom_data, top_data);
    if (input_scale_ != Dtype(1)) {
      this->device_->scal(count, input_scale_, top_data);
    }
    if (input_shift_ != Dtype(0)) {
      this->device_->add_scalar(count, input_shift_, top_data);
    }
    this->device_->template log<Dtype>(count, top_data, top_data);
  }
  if (base_scale_ != Dtype(1)) {
    this->device_->scal(count, base_scale_, top_data);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void LogLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                    const vector<Blob<MOtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<MItype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const int_tp count = bottom[0]->count();
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<const Dtype> top_diff = top[0]->gpu_diff();
  vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();

  this->device_->template copy<Dtype>(count, bottom_data, bottom_diff);
  if (input_scale_ != Dtype(1)) {
    this->device_->template scal<Dtype>(count, input_scale_, bottom_diff);
  }
  if (input_shift_ != Dtype(0)) {
    this->device_->add_scalar(count, input_shift_, bottom_diff);
  }
  this->device_->template powx<Dtype>(count, bottom_diff, Dtype(-1),
                                      bottom_diff);
  if (backward_num_scale_ != Dtype(1)) {
    this->device_->template scal<Dtype>(count, backward_num_scale_,
                                        bottom_diff);
  }
  this->device_->template mul<Dtype>(count, top_diff, bottom_diff, bottom_diff);
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(LogLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LogLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LogLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(LogLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LogLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LogLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
