#include <vector>

#include "caffe/layers/power_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void PowerLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  const int_tp count = bottom[0]->count();

  // Special case where we can ignore the input: scale or power is 0.
  if (diff_scale_ == Dtype(0)) {
    Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
    this->device_->set(count, value, top_data);
    return;
  }
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  this->device_->template copy<Dtype>(count, bottom_data, top_data);
  if (scale_ != Dtype(1)) {
    this->device_->template scal<Dtype>(count, scale_, top_data);
  }
  if (shift_ != Dtype(0)) {
    this->device_->template add_scalar<Dtype>(count, shift_, top_data);
  }
  if (power_ != Dtype(1)) {
    this->device_->template powx<Dtype>(count, top_data, power_, top_data);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void PowerLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                      const vector<Blob<MOtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
    const int_tp count = bottom[0]->count();
    vptr<const Dtype> top_diff = top[0]->gpu_diff();

    if (diff_scale_ == Dtype(0) || power_ == Dtype(1)) {
      this->device_->set(count, diff_scale_, bottom_diff);
    } else {
      vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
      // Compute dy/dx = scale * power * (shift + scale * X)^(power - 1)
      //               = diff_scale * Y / (shift + scale * X)
      if (power_ == Dtype(2)) {
        // Special case for Y = (shift + scale * X)^2
        //     -> dy/dx = 2 * scale * (shift + scale * X)
        //              = diff_scale * shift + diff_scale * scale * X
        this->device_->template axpby<Dtype>(count, diff_scale_ * scale_,
                       bottom_data, Dtype(0), bottom_diff);
        if (shift_ != Dtype(0)) {
          this->device_->template add_scalar<Dtype>(count, diff_scale_ * shift_,
                                                    bottom_diff);
        }
      } else if (shift_ == Dtype(0)) {
        // Special case for Y = (scale * X)^power
        //     -> dy/dx = scale * power * (scale * X)^(power - 1)
        //              = scale * power * (scale * X)^power * (scale * X)^(-1)
        //              = power * Y / X
        vptr<const Dtype> top_data = top[0]->gpu_data();
        this->device_->template div<Dtype>(count, top_data, bottom_data,
                                           bottom_diff);
        this->device_->template scal<Dtype>(count, power_, bottom_diff);
      } else {
        this->device_->template copy<Dtype>(count, bottom_data, bottom_diff);
        if (scale_ != Dtype(1)) {
          this->device_->template scal<Dtype>(count, scale_, bottom_diff);
        }
        if (shift_ != Dtype(0)) {
          this->device_->template add_scalar<Dtype>(count, shift_, bottom_diff);
        }
        vptr<const Dtype> top_data = top[0]->gpu_data();
        this->device_->template div<Dtype>(count, top_data, bottom_diff,
                                           bottom_diff);
        if (diff_scale_ != Dtype(1)) {
          this->device_->template scal<Dtype>(count, diff_scale_, bottom_diff);
        }
      }
    }
    this->device_->template mul<Dtype>(count, top_diff, bottom_diff,
                                       bottom_diff);
  }
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(PowerLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PowerLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PowerLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(PowerLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PowerLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PowerLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
