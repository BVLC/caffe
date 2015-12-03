#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void BatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int_tp num = bottom[0]->shape(0);
  int_tp spatial_dim = bottom[0]->count() / (channels_ * bottom[0]->shape(0));

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (bottom[0] != top[0]) {
      caffe_copy(bottom[0]->count(), bottom_data, top_data);
    }

    if (use_global_stats_) {
      // use the stored mean/variance estimates.
      const Dtype scale_factor =
          this->blobs_[2]->cpu_data()[0] == 0 ?
              0 : 1 / this->blobs_[2]->cpu_data()[0];
      caffe_gpu_scale(variance_.count(), scale_factor,
                      this->blobs_[0]->gpu_data(), mean_.mutable_gpu_data());
      caffe_gpu_scale(variance_.count(), scale_factor,
                      this->blobs_[1]->gpu_data(),
                      variance_.mutable_gpu_data());
    } else {
      // compute mean
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
                            1. / (num * spatial_dim), bottom_data,
                            spatial_sum_multiplier_.gpu_data(), 0.,
                            num_by_chans_.mutable_gpu_data());
      caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
                            num_by_chans_.gpu_data(),
                            batch_sum_multiplier_.gpu_data(), 0.,
                            mean_.mutable_gpu_data());
    }

    // subtract mean
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                          batch_sum_multiplier_.gpu_data(), mean_.gpu_data(),
                          0., num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
                          spatial_dim, 1, -1, num_by_chans_.gpu_data(),
                          spatial_sum_multiplier_.gpu_data(), 1., top_data);

    if (!use_global_stats_) {
      // compute variance using var(X) = E((X-EX)^2)
      caffe_gpu_powx(top[0]->count(), top_data, Dtype(2),
                     temp_.mutable_gpu_data());  // (X-EX)^2
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
                            1. / (num * spatial_dim), temp_.gpu_data(),
                            spatial_sum_multiplier_.gpu_data(), 0.,
                            num_by_chans_.mutable_gpu_data());
      caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
                            num_by_chans_.gpu_data(),
                            batch_sum_multiplier_.gpu_data(), 0.,
                            variance_.mutable_gpu_data());  // E((X_EX)^2)

      // compute and save moving average
      this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
      this->blobs_[2]->mutable_cpu_data()[0] += 1;
      caffe_gpu_axpby(mean_.count(), Dtype(1), mean_.gpu_data(),
                      moving_average_fraction_,
                      this->blobs_[0]->mutable_gpu_data());
      int_tp m = bottom[0]->count() / channels_;
      Dtype bias_correction_factor = m > 1 ? Dtype(m) / (m - 1) : 1;
      caffe_gpu_axpby(variance_.count(), bias_correction_factor,
                      variance_.gpu_data(), moving_average_fraction_,
                      this->blobs_[1]->mutable_gpu_data());
    }

    // normalize variance
    caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
    caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5),
                   variance_.mutable_gpu_data());

    // replicate variance to input size
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                          batch_sum_multiplier_.gpu_data(),
                          variance_.gpu_data(), 0.,
                          num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
                          spatial_dim, 1, 1., num_by_chans_.gpu_data(),
                          spatial_sum_multiplier_.gpu_data(), 0.,
                          temp_.mutable_gpu_data());
    caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
    // TODO(cdoersch): The caching is only needed because later in-place layers
    //                 might clobber the data.  Can we skip this if they won't?
    caffe_copy(x_norm_.count(), top_data, x_norm_.mutable_gpu_data());
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());

    if (bottom[0] != top[0]) {
      greentea_copy<Dtype>(bottom[0]->count(), (cl_mem) bottom_data, 0,
                           (cl_mem) top_data, 0, &ctx);
    }

    if (use_global_stats_) {
      // use the stored mean/variance estimates.
      const Dtype scale_factor =
          this->blobs_[2]->cpu_data()[0] == 0 ?
              0 : 1 / this->blobs_[2]->cpu_data()[0];
      greentea_gpu_scale<Dtype>(this->device_->id(), variance_.count(),
                                scale_factor,
                                (cl_mem) (this->blobs_[0]->gpu_data()), 0,
                                (cl_mem) (mean_.mutable_gpu_data()), 0);
      greentea_gpu_scale<Dtype>(this->device_->id(), variance_.count(),
                                scale_factor,
                                (cl_mem) (this->blobs_[1]->gpu_data()), 0,
                                (cl_mem) (variance_.mutable_gpu_data()), 0);
    } else {
      // compute mean
      greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans,
                               channels_ * num, spatial_dim,
                               1. / (num * spatial_dim), (cl_mem) bottom_data,
                               0, (cl_mem) (spatial_sum_multiplier_.gpu_data()),
                               0, 0.,
                               (cl_mem) (num_by_chans_.mutable_gpu_data()), 0);
      greentea_gpu_gemv<Dtype>(this->device_->id(), CblasTrans, num, channels_,
                               1., (cl_mem) (num_by_chans_.gpu_data()), 0,
                               (cl_mem) (batch_sum_multiplier_.gpu_data()), 0,
                               0., (cl_mem) (mean_.mutable_gpu_data()), 0);
    }

    // subtract mean
    greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans, CblasNoTrans,
                             num, channels_, 1, 1,
                             (cl_mem) (batch_sum_multiplier_.gpu_data()), 0,
                             (cl_mem) (mean_.gpu_data()), 0, 0.,
                             (cl_mem) (num_by_chans_.mutable_gpu_data()), 0);
    greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans, CblasNoTrans,
                             channels_ * num, spatial_dim, 1, -1,
                             (cl_mem) (num_by_chans_.gpu_data()), 0,
                             (cl_mem) (spatial_sum_multiplier_.gpu_data()), 0,
                             1., (cl_mem) top_data, 0);

    if (!use_global_stats_) {
      // compute variance using var(X) = E((X-EX)^2)
      greentea_gpu_powx<Dtype>(this->device_->id(), top[0]->count(),
                               (cl_mem) top_data, 0, Dtype(2),
                               (cl_mem) (temp_.mutable_gpu_data()), 0);
      // (X-EX)^2
      greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans,
                               channels_ * num, spatial_dim,
                               1. / (num * spatial_dim),
                               (cl_mem) (temp_.gpu_data()), 0,
                               (cl_mem) (spatial_sum_multiplier_.gpu_data()), 0,
                               0., (cl_mem) (num_by_chans_.mutable_gpu_data()),
                               0);
      greentea_gpu_gemv<Dtype>(this->device_->id(), CblasTrans, num, channels_,
                               1., (cl_mem) (num_by_chans_.gpu_data()), 0,
                               (cl_mem) (batch_sum_multiplier_.gpu_data()), 0,
                               0., (cl_mem) (variance_.mutable_gpu_data()), 0);
      // E((X_EX)^2)

      // compute and save moving average
      this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
      this->blobs_[2]->mutable_cpu_data()[0] += 1;
      greentea_gpu_axpby<Dtype>(this->device_->id(), mean_.count(), Dtype(1),
                                (cl_mem) (mean_.gpu_data()), 0,
                                moving_average_fraction_,
                                (cl_mem) (this->blobs_[0]->mutable_gpu_data()),
                                0);
      int_tp m = bottom[0]->count() / channels_;
      Dtype bias_correction_factor = m > 1 ? Dtype(m) / (m - 1) : 1;
      greentea_gpu_axpby<Dtype>(this->device_->id(), variance_.count(),
                                bias_correction_factor,
                                (cl_mem) (variance_.gpu_data()), 0,
                                moving_average_fraction_,
                                (cl_mem) (this->blobs_[1]->mutable_gpu_data()),
                                0);
    }

    // normalize variance
    greentea_gpu_add_scalar<Dtype>(this->device_->id(), variance_.count(), eps_,
                                   (cl_mem) (variance_.mutable_gpu_data()), 0);
    greentea_gpu_powx<Dtype>(this->device_->id(), variance_.count(),
                             (cl_mem) (variance_.gpu_data()), 0, Dtype(0.5),
                             (cl_mem) (variance_.mutable_gpu_data()), 0);

    // replicate variance to input size
    greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans, CblasNoTrans,
                             num, channels_, 1, 1,
                             (cl_mem) (batch_sum_multiplier_.gpu_data()), 0,
                             (cl_mem) (variance_.gpu_data()), 0, 0.,
                             (cl_mem) (num_by_chans_.mutable_gpu_data()), 0);
    greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans, CblasNoTrans,
                             channels_ * num, spatial_dim, 1, 1.,
                             (cl_mem) (num_by_chans_.gpu_data()), 0,
                             (cl_mem) (spatial_sum_multiplier_.gpu_data()), 0,
                             0., (cl_mem) (temp_.mutable_gpu_data()), 0);
    greentea_gpu_div<Dtype>(this->device_->id(), temp_.count(),
                            (cl_mem) top_data, 0, (cl_mem) (temp_.gpu_data()),
                            0, (cl_mem) top_data, 0);
    // TODO(cdoersch): The caching is only needed because later in-place layers
    //                 might clobber the data.  Can we skip this if they won't?
    greentea_copy<Dtype>(x_norm_.count(), (cl_mem) top_data, 0,
                         (cl_mem) (x_norm_.mutable_gpu_data()), 0, &ctx);
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void BatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff;

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (bottom[0] != top[0]) {
      top_diff = top[0]->gpu_diff();
    } else {
      caffe_copy(x_norm_.count(), top[0]->gpu_diff(),
                 x_norm_.mutable_gpu_diff());
      top_diff = x_norm_.gpu_diff();
    }
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (use_global_stats_) {
      caffe_gpu_div(temp_.count(), top_diff, temp_.gpu_data(), bottom_diff);
      return;
    }
    const Dtype* top_data = x_norm_.gpu_data();
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
    caffe_gpu_mul<Dtype>(temp_.count(), top_data, top_diff, bottom_diff);
    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
                          bottom_diff, spatial_sum_multiplier_.gpu_data(), 0.,
                          num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
                          num_by_chans_.gpu_data(),
                          batch_sum_multiplier_.gpu_data(), 0.,
                          mean_.mutable_gpu_data());

    // reshape (broadcast) the above
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                          batch_sum_multiplier_.gpu_data(), mean_.gpu_data(),
                          0., num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
                          spatial_dim, 1, 1., num_by_chans_.gpu_data(),
                          spatial_sum_multiplier_.gpu_data(), 0., bottom_diff);

    // sum(dE/dY \cdot Y) \cdot Y
    caffe_gpu_mul<Dtype>(temp_.count(), top_data, bottom_diff, bottom_diff);

    // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
                          top_diff, spatial_sum_multiplier_.gpu_data(), 0.,
                          num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
                          num_by_chans_.gpu_data(),
                          batch_sum_multiplier_.gpu_data(), 0.,
                          mean_.mutable_gpu_data());
    // reshape (broadcast) the above to make
    // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                          batch_sum_multiplier_.gpu_data(), mean_.gpu_data(),
                          0., num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
                          spatial_dim, 1, 1., num_by_chans_.gpu_data(),
                          spatial_sum_multiplier_.gpu_data(), 1., bottom_diff);

    // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
    caffe_gpu_axpby<Dtype>(temp_.count(), Dtype(1), top_diff,
                           Dtype(-1. / (num * spatial_dim)), bottom_diff);

    // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
    // pass.
    caffe_gpu_div<Dtype>(temp_.count(), bottom_diff, temp_.gpu_data(),
                         bottom_diff);
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());

    if (bottom[0] != top[0]) {
      top_diff = top[0]->gpu_diff();
    } else {
      greentea_copy<Dtype>(x_norm_.count(), (cl_mem) (top[0]->gpu_diff()), 0,
                           (cl_mem) (x_norm_.mutable_gpu_diff()), 0, &ctx);
      top_diff = x_norm_.gpu_diff();
    }
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (use_global_stats_) {
      greentea_gpu_div<Dtype>(this->device_->id(), temp_.count(),
                              (cl_mem) top_diff, 0, (cl_mem) (temp_.gpu_data()),
                              0, (cl_mem) bottom_diff, 0);
      return;
    }
    const Dtype* top_data = x_norm_.gpu_data();
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
    greentea_gpu_mul<Dtype>(this->device_->id(), temp_.count(),
                            (cl_mem) top_data, 0, (cl_mem) top_diff, 0,
                            (cl_mem) bottom_diff, 0);
    greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans, channels_ * num,
                             spatial_dim, 1., (cl_mem) bottom_diff, 0,
                             (cl_mem) (spatial_sum_multiplier_.gpu_data()), 0,
                             0., (cl_mem) (num_by_chans_.mutable_gpu_data()),
                             0);
    greentea_gpu_gemv<Dtype>(this->device_->id(), CblasTrans, num, channels_,
                             1., (cl_mem) (num_by_chans_.gpu_data()), 0,
                             (cl_mem) (batch_sum_multiplier_.gpu_data()), 0, 0.,
                             (cl_mem) (mean_.mutable_gpu_data()), 0);

    // reshape (broadcast) the above
    greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans, CblasNoTrans,
                             num, channels_, 1, 1,
                             (cl_mem) (batch_sum_multiplier_.gpu_data()), 0,
                             (cl_mem) (mean_.gpu_data()), 0, 0.,
                             (cl_mem) (num_by_chans_.mutable_gpu_data()), 0);
    greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans, CblasNoTrans,
                             channels_ * num, spatial_dim, 1, 1.,
                             (cl_mem) (num_by_chans_.gpu_data()), 0,
                             (cl_mem) (spatial_sum_multiplier_.gpu_data()), 0,
                             0., (cl_mem) bottom_diff, 0);

    // sum(dE/dY \cdot Y) \cdot Y
    greentea_gpu_mul<Dtype>(this->device_->id(), temp_.count(),
                            (cl_mem) top_data, 0, (cl_mem) bottom_diff, 0,
                            (cl_mem) bottom_diff, 0);

    // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
    greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans, channels_ * num,
                             spatial_dim, 1., (cl_mem) top_diff, 0,
                             (cl_mem) (spatial_sum_multiplier_.gpu_data()), 0,
                             0., (cl_mem) (num_by_chans_.mutable_gpu_data()),
                             0);
    greentea_gpu_gemv<Dtype>(this->device_->id(), CblasTrans, num, channels_,
                             1., (cl_mem) (num_by_chans_.gpu_data()), 0,
                             (cl_mem) (batch_sum_multiplier_.gpu_data()), 0, 0.,
                             (cl_mem) (mean_.mutable_gpu_data()), 0);
    // reshape (broadcast) the above to make
    // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
    greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans, CblasNoTrans,
                             num, channels_, 1, 1,
                             (cl_mem) (batch_sum_multiplier_.gpu_data()), 0,
                             (cl_mem) (mean_.gpu_data()), 0, 0.,
                             (cl_mem) (num_by_chans_.mutable_gpu_data()), 0);
    greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans, CblasNoTrans,
                             num * channels_, spatial_dim, 1, 1.,
                             (cl_mem) (num_by_chans_.gpu_data()), 0,
                             (cl_mem) (spatial_sum_multiplier_.gpu_data()), 0,
                             1., (cl_mem) bottom_diff, 0);

    // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
    greentea_gpu_axpby<Dtype>(this->device_->id(), temp_.count(), Dtype(1),
                              (cl_mem) top_diff, 0,
                              Dtype(-1. / (num * spatial_dim)),
                              (cl_mem) bottom_diff, 0);

    // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
    // pass.
    greentea_gpu_div<Dtype>(this->device_->id(), temp_.count(),
                            (cl_mem) bottom_diff, 0,
                            (cl_mem) (temp_.gpu_data()), 0,
                            (cl_mem) bottom_diff, 0);
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchNormLayer);

}  // namespace caffe
