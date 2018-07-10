#include <vector>

#include "caffe/layers/exp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void ExpLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                    const vector<Blob<MItype>*>& bottom,
                                    const vector<Blob<MOtype>*>& top) {
  const int_tp count = bottom[0]->count();
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();

  if (inner_scale_ == Dtype(1)) {
    this->device_->template exp<Dtype>(count, bottom_data, top_data);
  } else {
    this->device_->template scale<Dtype>(count, inner_scale_,
                                         bottom_data, top_data);
    this->device_->template exp<Dtype>(count, top_data, top_data);
  }
  if (outer_scale_ != Dtype(1)) {
    this->device_->template scal<Dtype>(count, outer_scale_, top_data);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void ExpLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                    const vector<Blob<MOtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<MItype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const int_tp count = bottom[0]->count();
  vptr<const Dtype> top_data = top[0]->gpu_data();
  vptr<const Dtype> top_diff = top[0]->gpu_diff();
  vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();

  this->device_->template mul<Dtype>(count, top_data, top_diff, bottom_diff);
  if (inner_scale_ != Dtype(1)) {
    this->device_->template scal<Dtype>(count, inner_scale_, bottom_diff);
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ExpLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ExpLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ExpLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ExpLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ExpLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ExpLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
