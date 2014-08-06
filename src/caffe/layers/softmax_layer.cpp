// Copyright 2014 BVLC and contributors.
//
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  sum_multiplier_.Reshape(1, bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  for (int i = 0; i < sum_multiplier_.count(); ++i) {
    multiplier_data[i] = 1.;
  }
  scale_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
Dtype SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // we need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < num; ++i) {
    scale_data[i] = bottom_data[i*dim];
    for (int j = 0; j < dim; ++j) {
      scale_data[i] = std::max(scale_data[i], bottom_data[i * dim + j]);
    }
  }
  // subtraction
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
    scale_data, sum_multiplier_.cpu_data(), 1., top_data);
  // Perform exponentiation
  caffe_exp<Dtype>(num * dim, top_data, top_data);
  // sum after exp
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_data,
      sum_multiplier_.cpu_data(), 0., scale_data);
  // Do division
  for (int i = 0; i < num; ++i) {
    caffe_scal<Dtype>(dim, Dtype(1.) / scale_data[i], top_data + i * dim);
  }
  return Dtype(0);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int num = top[0]->num();
  int dim = top[0]->count() / top[0]->num();
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
  for (int i = 0; i < num; ++i) {
    scale_data[i] = caffe_cpu_dot<Dtype>(dim, top_diff + i * dim,
        top_data + i * dim);
  }
  // subtraction
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      scale_data, sum_multiplier_.cpu_data(), 1., bottom_diff);
  // elementwise multiplication
  caffe_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);


}  // namespace caffe
