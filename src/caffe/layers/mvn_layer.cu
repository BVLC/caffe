#include <vector>

#include "caffe/layers/mvn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MVNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;

  // subtract mean
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());  // EX
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
      temp_.mutable_gpu_data());
  caffe_gpu_add(temp_.count(), bottom_data, temp_.gpu_data(),
      top_data);  // X-EX

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

    caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
          temp_.mutable_gpu_data());

    caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(MVNLayer);


}  // namespace caffe
