#include <vector>

#include "caffe/layers/mvn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void MVNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int_tp num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int_tp dim = bottom[0]->count() / num;

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // subtract mean
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
                          sum_multiplier_.gpu_data(), 0.,
                          mean_.mutable_gpu_data());  // EX
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
                          mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
                          temp_.mutable_gpu_data());
    // X-EX
    caffe_gpu_add(temp_.count(), bottom_data, temp_.gpu_data(), top_data);

    if (this->layer_param_.mvn_param().normalize_variance()) {
      // compute variance using var(X) = E((X-EX)^2)
      caffe_gpu_powx(bottom[0]->count(), top_data, Dtype(2),
                     temp_.mutable_gpu_data());  // (X-EX)^2
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.gpu_data(),
                            sum_multiplier_.gpu_data(), 0.,
                            variance_.mutable_gpu_data());  // E((X-EX)^2)

      // normalize variance
      caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5),
                     variance_.mutable_gpu_data());

      caffe_gpu_add_scalar(variance_.count(), eps_,
                           variance_.mutable_gpu_data());

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
                            variance_.gpu_data(), sum_multiplier_.gpu_data(),
                            0., temp_.mutable_gpu_data());

      caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    // subtract mean
    greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans, num, dim,
                             1. / dim, (cl_mem) (bottom_data), 0,
                             (cl_mem) (sum_multiplier_.gpu_data()), 0, 0.,
                             (cl_mem) (mean_.mutable_gpu_data()), 0);  // EX
    greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans, CblasNoTrans,
                             num, dim, 1, -1., (cl_mem) (mean_.gpu_data()), 0,
                             (cl_mem) (sum_multiplier_.gpu_data()), 0, 0.,
                             (cl_mem) (temp_.mutable_gpu_data()), 0);
    greentea_gpu_add<Dtype>(this->device_->id(), temp_.count(),
                            (cl_mem) (bottom_data), 0,
                            (cl_mem) (temp_.gpu_data()), 0, (cl_mem) (top_data),
                            0);  // X-EX

    if (this->layer_param_.mvn_param().normalize_variance()) {
      // compute variance using var(X) = E((X-EX)^2)
      // (X-EX)^2
      greentea_gpu_powx<Dtype>(this->device_->id(), bottom[0]->count(),
                               (cl_mem) (top_data), 0, Dtype(2),
                               (cl_mem) (temp_.mutable_gpu_data()), 0);
      // E((X-EX)^2)
      greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans, num, dim,
                               1. / dim, (cl_mem) (temp_.gpu_data()), 0,
                               (cl_mem) (sum_multiplier_.gpu_data()), 0, 0.,
                               (cl_mem) (variance_.mutable_gpu_data()), 0);

      // normalize variance
      greentea_gpu_powx<Dtype>(this->device_->id(), variance_.count(),
                               (cl_mem) (variance_.gpu_data()), 0, Dtype(0.5),
                               (cl_mem) (variance_.mutable_gpu_data()), 0);

      greentea_gpu_add_scalar<Dtype>(this->device_->id(), variance_.count(),
                                     eps_,
                                     (cl_mem) (variance_.mutable_gpu_data()),
                                     0);

      greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans, CblasNoTrans,
                               num, dim, 1, 1., (cl_mem) (variance_.gpu_data()),
                               0, (cl_mem) (sum_multiplier_.gpu_data()), 0, 0.,
                               (cl_mem) (temp_.mutable_gpu_data()), 0);

      greentea_gpu_div<Dtype>(this->device_->id(), temp_.count(),
                              (cl_mem) (top_data), 0,
                              (cl_mem) (temp_.gpu_data()), 0,
                              (cl_mem) (top_data), 0);
    }
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void MVNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  int_tp num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int_tp dim = bottom[0]->count() / num;

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (this->layer_param_.mvn_param().normalize_variance()) {
      caffe_gpu_mul(temp_.count(), top_data, top_diff, bottom_diff);
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., bottom_diff,
                            sum_multiplier_.gpu_data(), 0.,
                            mean_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
                            mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
                            bottom_diff);
      caffe_gpu_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

      caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_diff,
                            sum_multiplier_.gpu_data(), 0.,
                            mean_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
                            mean_.gpu_data(), sum_multiplier_.gpu_data(), 1.,
                            bottom_diff);

      caffe_gpu_axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1. / dim),
                      bottom_diff);

      // put the squares of bottom into temp_
      caffe_gpu_powx(temp_.count(), bottom_data, Dtype(2),
                     temp_.mutable_gpu_data());

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
                            variance_.gpu_data(), sum_multiplier_.gpu_data(),
                            0., temp_.mutable_gpu_data());

      caffe_gpu_div(temp_.count(), bottom_diff, temp_.gpu_data(), bottom_diff);
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, top_diff,
                            sum_multiplier_.gpu_data(), 0.,
                            mean_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
                            mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
                            temp_.mutable_gpu_data());
      caffe_gpu_add<Dtype>(temp_.count(), top_diff, temp_.gpu_data(),
                           bottom_diff);
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    if (this->layer_param_.mvn_param().normalize_variance()) {
      greentea_gpu_mul<Dtype>(this->device_->id(), temp_.count(),
                              (cl_mem) top_data, 0, (cl_mem) top_diff, 0,
                              (cl_mem) bottom_diff, 0);
      greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans, num,
                               dim, 1., (cl_mem) bottom_diff, 0,
                               (cl_mem) (sum_multiplier_.gpu_data()), 0, 0.,
                               (cl_mem) (mean_.mutable_gpu_data()), 0);
      greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                               CblasNoTrans, num, dim, 1, 1.,
                               (cl_mem) (mean_.gpu_data()), 0,
                               (cl_mem) (sum_multiplier_.gpu_data()), 0, 0.,
                               (cl_mem) bottom_diff, 0);
      greentea_gpu_mul<Dtype>(this->device_->id(), temp_.count(),
                              (cl_mem) top_data, 0, (cl_mem) bottom_diff, 0,
                              (cl_mem) bottom_diff, 0);

      greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans, num,
                               dim, 1., (cl_mem) top_diff, 0,
                               (cl_mem) (sum_multiplier_.gpu_data()), 0, 0.,
                               (cl_mem) (mean_.mutable_gpu_data()), 0);
      greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                               CblasNoTrans, num, dim, 1, 1.,
                               (cl_mem) (mean_.gpu_data()), 0,
                               (cl_mem) (sum_multiplier_.gpu_data()), 0, 1.,
                               (cl_mem) bottom_diff, 0);

      greentea_gpu_axpby<Dtype>(this->device_->id(), temp_.count(),
                                Dtype(1), (cl_mem) top_diff, 0,
                                Dtype(-1. / dim), (cl_mem) bottom_diff, 0);

      // put the squares of bottom into temp_
      greentea_gpu_powx<Dtype>(this->device_->id(), temp_.count(),
                               (cl_mem) bottom_data, 0, Dtype(2),
                               (cl_mem) (temp_.mutable_gpu_data()), 0);

      greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                               CblasNoTrans, num, dim, 1, 1.,
                               (cl_mem) (variance_.gpu_data()), 0,
                               (cl_mem) (sum_multiplier_.gpu_data()), 0, 0.,
                               (cl_mem) (temp_.mutable_gpu_data()), 0);

      greentea_gpu_div<Dtype>(this->device_->id(), temp_.count(),
                              (cl_mem) bottom_diff, 0,
                              (cl_mem) (temp_.gpu_data()), 0,
                              (cl_mem) bottom_diff, 0);
    } else {
      greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans, num,
                               dim, 1. / dim, (cl_mem) top_diff, 0,
                               (cl_mem) (sum_multiplier_.gpu_data()), 0, 0.,
                               (cl_mem) (mean_.mutable_gpu_data()), 0);
      greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                               CblasNoTrans, num, dim, 1, -1.,
                               (cl_mem) (mean_.gpu_data()), 0,
                               (cl_mem) (sum_multiplier_.gpu_data()), 0, 0.,
                               (cl_mem) (temp_.mutable_gpu_data()), 0);
      greentea_gpu_add<Dtype>(this->device_->id(), temp_.count(),
                              (cl_mem) top_diff, 0, (cl_mem) (temp_.gpu_data()),
                              0, (cl_mem) (bottom_diff), 0);
    }
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MVNLayer);

}  // namespace caffe
