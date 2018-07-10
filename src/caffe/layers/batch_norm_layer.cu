#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void BatchNormLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                const vector<Blob<MItype>*>& bottom,
                                const vector<Blob<MOtype>*>& top) {
  vptr<const MItype> bottom_data = bottom[0]->gpu_data();
  vptr<MOtype> top_data = top[0]->mutable_gpu_data();
  int_tp num = bottom[0]->shape(0);
  int_tp spatial_dim = bottom[0]->count() / (channels_ * bottom[0]->shape(0));

  if (bottom[0] != top[0]) {
      this->device_->copy(bottom[0]->count(), bottom_data, top_data);
    }
    if (use_global_stats_) {
      // use the stored mean/variance estimates.
      const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
          0 : 1 / this->blobs_[2]->cpu_data()[0];
      this->device_->scale(variance_.count(), scale_factor,
          this->blobs_[0]->gpu_data(), mean_.mutable_gpu_data());
      this->device_->scale(variance_.count(), scale_factor,
          this->blobs_[1]->gpu_data(), variance_.mutable_gpu_data());
    } else {
      // compute mean
      this->device_->template gemv<Dtype>(CblasNoTrans,
          channels_ * num, spatial_dim, 1. / (num * spatial_dim), bottom_data,
          spatial_sum_multiplier_.gpu_data(), 0.,
          num_by_chans_.mutable_gpu_data());
      this->device_->template gemv<Dtype>(CblasTrans, num, channels_, 1.,
          num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
          mean_.mutable_gpu_data());
    }

    // subtract mean
    this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
        channels_, 1, 1, batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
        num_by_chans_.mutable_gpu_data());
    this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans,
        channels_ * num, spatial_dim, 1, -1, num_by_chans_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), 1., top_data);

    if (!use_global_stats_) {
      // compute variance using var(X) = E((X-EX)^2)
      this->device_->template mul(top[0]->count(), top[0]->gpu_data(),
                     top[0]->gpu_data(), temp_.mutable_gpu_data());  // (X-EX)^2
      this->device_->template gemv<Dtype>(CblasNoTrans, channels_ * num,
          spatial_dim, 1. / (num * spatial_dim), temp_.gpu_data(),
          spatial_sum_multiplier_.gpu_data(), 0.,
          num_by_chans_.mutable_gpu_data());
      this->device_->template gemv<Dtype>(CblasTrans, num, channels_, Dtype(1.),
          num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(),
          Dtype(0.), variance_.mutable_gpu_data());  // E((X_EX)^2)

      // compute and save moving average
      this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
      this->blobs_[2]->mutable_cpu_data()[0] += 1;
      this->device_->axpby(mean_.count(), Dtype(1), mean_.gpu_data(),
          moving_average_fraction_, this->blobs_[0]->mutable_gpu_data());
      int_tp m = bottom[0]->count()/channels_;
      Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
      this->device_->axpby(variance_.count(), bias_correction_factor,
                           variance_.gpu_data(), moving_average_fraction_,
                           this->blobs_[1]->mutable_gpu_data());
    }

    // normalize variance
    this->device_->add_scalar(variance_.count(),
                              eps_, variance_.mutable_gpu_data());
    this->device_->sqrt(variance_.count(), variance_.gpu_data(),
                        variance_.mutable_gpu_data());

    // replicate variance to input size
    this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
        channels_, 1, 1, batch_sum_multiplier_.gpu_data(), variance_.gpu_data(),
        0., num_by_chans_.mutable_gpu_data());
    this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans,
        channels_ * num, spatial_dim, 1, 1., num_by_chans_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
    this->device_->template div<Dtype>(temp_.count(), top_data,
                                       temp_.gpu_data(), top_data);
    // TODO(cdoersch): The caching is only needed
    //                 because later in-place layers might clobber the data.
    //                 Can we skip this if they won't?
    this->device_->template copy<Dtype>(x_norm_.count(),
                                        top_data, x_norm_.mutable_gpu_data());
}

template<typename Dtype, typename MItype, typename MOtype>
void BatchNormLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                         const vector<Blob<MOtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<MItype>*>& bottom) {
  vptr<const MOtype> top_diff;

  if (bottom[0] != top[0]) {
    top_diff = top[0]->gpu_diff();
  } else {
    this->device_->copy(x_norm_.count(), top[0]->gpu_diff(),
                        x_norm_.mutable_gpu_diff());
    top_diff = x_norm_.gpu_diff();
  }
  vptr<MItype> bottom_diff = bottom[0]->mutable_gpu_diff();
  if (use_global_stats_) {
    this->device_->div(temp_.count(), top_diff, temp_.gpu_data(), bottom_diff);
    return;
  }
  vptr<const Dtype> top_data = x_norm_.gpu_data();
  int_tp num = bottom[0]->shape()[0];
  int_tp spatial_dim = bottom[0]->count() / (channels_ * bottom[0]->shape(0));
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  this->device_->template mul<Dtype>(temp_.count(), top_data,
                                     top_diff, bottom_diff);
  this->device_->template gemv<Dtype>(CblasNoTrans, channels_ * num,
                        spatial_dim, 1., bottom_diff,
                        spatial_sum_multiplier_.gpu_data(), 0.,
                        num_by_chans_.mutable_gpu_data());
  this->device_->template gemv<Dtype>(CblasTrans, num, channels_, 1.,
                        num_by_chans_.gpu_data(),
                        batch_sum_multiplier_.gpu_data(), 0.,
                        mean_.mutable_gpu_data());

  // reshape (broadcast) the above
  this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
                        channels_, 1, 1, batch_sum_multiplier_.gpu_data(),
                        mean_.gpu_data(), 0., num_by_chans_.mutable_gpu_data());
  this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                        channels_ * num, spatial_dim, 1, 1.,
                        num_by_chans_.gpu_data(),
                        spatial_sum_multiplier_.gpu_data(), 0., bottom_diff);

  // sum(dE/dY \cdot Y) \cdot Y
  this->device_->template mul<Dtype>(temp_.count(), top_data, bottom_diff,
                                     bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  this->device_->template gemv<Dtype>(CblasNoTrans, channels_ * num,
                        spatial_dim, 1., top_diff,
                        spatial_sum_multiplier_.gpu_data(), 0.,
                        num_by_chans_.mutable_gpu_data());
  this->device_->template gemv<Dtype>(CblasTrans, num, channels_, 1.,
                        num_by_chans_.gpu_data(),
                        batch_sum_multiplier_.gpu_data(), 0.,
                        mean_.mutable_gpu_data());
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
                        channels_, 1, 1, batch_sum_multiplier_.gpu_data(),
                        mean_.gpu_data(), 0., num_by_chans_.mutable_gpu_data());
  this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                        num * channels_, spatial_dim, 1, 1.,
                        num_by_chans_.gpu_data(),
                        spatial_sum_multiplier_.gpu_data(), 1., bottom_diff);

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  this->device_->template axpby<Dtype>(temp_.count(), Dtype(1), top_diff,
                         Dtype(-1. / (num * spatial_dim)), bottom_diff);

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  this->device_->template div<Dtype>(temp_.count(), bottom_diff,
                                     temp_.gpu_data(), bottom_diff);
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(BatchNormLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BatchNormLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BatchNormLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(BatchNormLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BatchNormLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BatchNormLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
