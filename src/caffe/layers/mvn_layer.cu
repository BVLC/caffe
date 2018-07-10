#include <vector>

#include "caffe/layers/mvn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void MVNLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                    const vector<Blob<MItype>*>& bottom,
                                    const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  int_tp num;
  if (this->layer_param_.mvn_param().across_channels()) {
    num = bottom[0]->num();
  } else {
    num = bottom[0]->num() * bottom[0]->channels();
  }

  int_tp dim = bottom[0]->count() / num;

  // subtract mean
  this->device_->template gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim,
                 bottom_data, sum_multiplier_.gpu_data(), 0.,
                 mean_.mutable_gpu_data());  // EX
  this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,
                 -1., mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
                 temp_.mutable_gpu_data());
  // X-EX
  this->device_->template add<Dtype>(temp_.count(), bottom_data,
                                     temp_.gpu_data(), top_data);

  if (this->layer_param_.mvn_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    this->device_->template powx<Dtype>(bottom[0]->count(), top_data, Dtype(2),
                   temp_.mutable_gpu_data());  // (X-EX)^2
    this->device_->template gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim,
                   temp_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
                   variance_.mutable_gpu_data());  // E((X-EX)^2)

    // normalize variance
    this->device_->template powx(variance_.count(), variance_.gpu_data(),
                   Dtype(0.5), variance_.mutable_gpu_data());

    this->device_->template add_scalar(variance_.count(), eps_,
                                       variance_.mutable_gpu_data());

    this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,
                   1., variance_.gpu_data(), sum_multiplier_.gpu_data(),
                   0., temp_.mutable_gpu_data());

    this->device_->template div<Dtype>(temp_.count(), top_data,
                                       temp_.gpu_data(), top_data);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void MVNLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                    const vector<Blob<MOtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<MItype>*>& bottom) {
  vptr<const Dtype> top_diff = top[0]->gpu_diff();
  vptr<const Dtype> top_data = top[0]->gpu_data();
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();

  int_tp num;
  if (this->layer_param_.mvn_param().across_channels()) {
    num = bottom[0]->num();
  } else {
    num = bottom[0]->num() * bottom[0]->channels();
  }

  int_tp dim = bottom[0]->count() / num;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    this->device_->template mul<Dtype>(temp_.count(), top_data, top_diff,
                                       bottom_diff);
    this->device_->template gemv<Dtype>(CblasNoTrans, num, dim, 1., bottom_diff,
                   sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
    this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,
                   1., mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
                   bottom_diff);
    this->device_->template mul<Dtype>(temp_.count(), top_data, bottom_diff,
                                       bottom_diff);

    this->device_->template gemv<Dtype>(CblasNoTrans, num, dim, 1., top_diff,
                   sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
    this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,
                   1., mean_.gpu_data(), sum_multiplier_.gpu_data(), 1.,
                   bottom_diff);

    this->device_->template axpby<Dtype>(temp_.count(), Dtype(1), top_diff,
                                         Dtype(-1. / dim), bottom_diff);

    // put the squares of bottom into temp_
    this->device_->template powx<Dtype>(temp_.count(), bottom_data, Dtype(2),
                   temp_.mutable_gpu_data());

    this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,
                   1., variance_.gpu_data(), sum_multiplier_.gpu_data(),
                   0., temp_.mutable_gpu_data());

    this->device_->template div<Dtype>(temp_.count(), bottom_diff,
                                       temp_.gpu_data(), bottom_diff);
  } else {
    this->device_->template gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim,
                   top_diff, sum_multiplier_.gpu_data(), 0.,
                   mean_.mutable_gpu_data());
    this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,
                   -1., mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
                   temp_.mutable_gpu_data());
    this->device_->template add<Dtype>(temp_.count(), top_diff,
                                       temp_.gpu_data(), bottom_diff);
  }
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(MVNLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MVNLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MVNLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(MVNLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MVNLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MVNLayer, Backward_gpu,
                                  (double), (double), (double));
}  // namespace caffe
