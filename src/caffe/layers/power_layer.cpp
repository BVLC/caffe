#include <vector>

#include "caffe/layers/power_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void PowerLayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  NeuronLayer<Dtype, MItype, MOtype>::LayerSetUp(bottom, top);
  power_ = this->layer_param_.power_param().power();
  scale_ = this->layer_param_.power_param().scale();
  shift_ = this->layer_param_.power_param().shift();
  diff_scale_ = power_  * scale_;
  this->InitializeQuantizers(bottom, top);
}

// Compute Y = (shift + scale * X)^power
template<typename Dtype, typename MItype, typename MOtype>
void PowerLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int_tp count = bottom[0]->count();
  // Special case where we can ignore the input: scale or power is 0.
  if (diff_scale_ == Dtype(0)) {
    Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
    caffe_set(count, value, top_data);
    return;
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  caffe_copy(count, bottom_data, top_data);
  if (scale_ != Dtype(1)) {
    caffe_scal(count, scale_, top_data);
  }
  if (shift_ != Dtype(0)) {
    caffe_add_scalar(count, shift_, top_data);
  }
  if (power_ != Dtype(1)) {
    caffe_powx(count, top_data, power_, top_data);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void PowerLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int_tp count = bottom[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    if (diff_scale_ == Dtype(0) || power_ == Dtype(1)) {
      caffe_set(count, diff_scale_, bottom_diff);
    } else {
      const Dtype* bottom_data = bottom[0]->cpu_data();
      // Compute dy/dx = scale * power * (shift + scale * X)^(power - 1)
      //               = diff_scale * Y / (shift + scale * X)
      if (power_ == Dtype(2)) {
        // Special case for Y = (shift + scale * X)^2
        //     -> dy/dx = 2 * scale * (shift + scale * X)
        //              = diff_scale * shift + diff_scale * scale * X
        caffe_axpby(count, Dtype(diff_scale_ * scale_), bottom_data,
            Dtype(0), bottom_diff);
        if (shift_ != Dtype(0)) {
          caffe_add_scalar(count, Dtype(diff_scale_ * shift_), bottom_diff);
        }
      } else if (shift_ == Dtype(0)) {
        // Special case for Y = (scale * X)^power
        //     -> dy/dx = scale * power * (scale * X)^(power - 1)
        //              = scale * power * (scale * X)^power * (scale * X)^(-1)
        //              = power * Y / X
        const Dtype* top_data = top[0]->cpu_data();
        caffe_div(count, top_data, bottom_data, bottom_diff);
        caffe_scal(count, power_, bottom_diff);
      } else {
        caffe_copy(count, bottom_data, bottom_diff);
        if (scale_ != Dtype(1)) {
          caffe_scal(count, scale_, bottom_diff);
        }
        if (shift_ != Dtype(0)) {
          caffe_add_scalar(count, shift_, bottom_diff);
        }
        const Dtype* top_data = top[0]->cpu_data();
        caffe_div<Dtype>(count, top_data, bottom_diff, bottom_diff);
        if (diff_scale_ != Dtype(1)) {
          caffe_scal(count, diff_scale_, bottom_diff);
        }
      }
    }
    if (diff_scale_ != Dtype(0)) {
      caffe_mul(count, top_diff, bottom_diff, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PowerLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(PowerLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(PowerLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(PowerLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(Power);
REGISTER_LAYER_CLASS_INST(Power, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(Power, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(Power, (double), (double), (double));


}  // namespace caffe
