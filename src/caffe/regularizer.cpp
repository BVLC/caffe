// Copyright 2014 kloudkl@github

#include <cmath>  // for std::abs

#include "caffe/proto/caffe.pb.h"
#include "caffe/regularizer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
Dtype Regularizer<Dtype>::Regularize(Blob<Dtype>* bottom) {
  Dtype penalty = 0;
  if (Caffe::mode() == Caffe::CPU) {
    penalty = Regularize_cpu(bottom);
  } else if (Caffe::mode() == Caffe::GPU) {
    penalty = Regularize_gpu(bottom);
  } else {
    LOG(FATAL)<< "Unknown mode: " << Caffe::mode();
  }
  return penalty;
}

template<typename Dtype>
Dtype L1Regularizer<Dtype>::Regularize_cpu(Blob<Dtype>* bottom) {
  if (this->coeff_ == 0) {
    return Dtype(0.);
  }
  const Dtype* data = bottom->cpu_data();
  Dtype* diff = bottom->mutable_cpu_diff();
  int count = bottom->count();
  for (int c = 0; c < count; ++c) {
    diff[c] += this->coeff_ * caffe_sign<Dtype>(data[c]);
  }
  Dtype penalty = caffe_cpu_asum(count, data);
  return this->coeff_ * penalty;
}

template<typename Dtype>
Dtype L2Regularizer<Dtype>::Regularize_cpu(Blob<Dtype>* bottom) {
  if (this->coeff_ == 0) {
    return Dtype(0);
  }
  const Dtype* data = bottom->cpu_data();
  Dtype* diff = bottom->mutable_cpu_diff();
  int count = bottom->count();
  caffe_axpy<Dtype>(count, this->coeff_ * 2., data, diff);
  Dtype penalty = caffe_cpu_dot<Dtype>(count, data, data);
  return this->coeff_ * penalty;
}

template<typename Dtype>
Dtype MaxNormRegularizer<Dtype>::Regularize_cpu(Blob<Dtype>* bottom) {
  if (this->coeff_ == 0) {
    return Dtype(0);
  }
  const Dtype* data = bottom->cpu_data();
  Dtype* diff = bottom->mutable_cpu_diff();
  int count = bottom->count();
  Dtype penalty = 0;
  // TODO: Implement MaxNormRegularizer::Regularize_cpu
  return this->coeff_ * penalty;
}

template<typename Dtype>
Regularizer<Dtype>* GetRegularizer(const RegularizerParameter& param) {
  const RegularizerParameter_RegularizerType type = param.type();
  if (type == REG_TYPE(L1)) {
    return new L1Regularizer<Dtype>(param);
  } else if (type == REG_TYPE(L2)) {
    return new L2Regularizer<Dtype>(param);
  } else if (type == REG_TYPE(MAX_NORM)) {
    return new MaxNormRegularizer<Dtype>(param);
  } else {
    LOG(FATAL) << "Unknown regularizer type: " << type;
  }
  // just to suppress old compiler warnings.
  return (Regularizer<Dtype>*) (NULL);
}

template Regularizer<float>* GetRegularizer<float>(
    const RegularizerParameter& param);
template Regularizer<double>* GetRegularizer<double>(
    const RegularizerParameter& param);

INSTANTIATE_CLASS(Regularizer);
INSTANTIATE_CLASS(L1Regularizer);
INSTANTIATE_CLASS(L2Regularizer);
INSTANTIATE_CLASS(MaxNormRegularizer);

}  // namespace caffe
