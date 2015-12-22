#include <vector>

<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/common_layers.hpp"
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/layers/mvn_layer.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/layers/mvn_layer.hpp"
>>>>>>> BVLC/master
#include "caffe/util/math_functions.hpp"
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
=======
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
>>>>>>> BVLC/device-abstraction
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
<<<<<<< HEAD
=======
=======
>>>>>>> pod/device/blob.hpp
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> pod/device/blob.hpp
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
>>>>>>> BVLC/device-abstraction
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
>>>>>>> BVLC/device-abstraction
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
>>>>>>> BVLC/device-abstraction
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
=======
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
>>>>>>> device-abstraction

namespace caffe {

template <typename Dtype>
void MVNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  mean_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  variance_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  temp_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> device-abstraction
  sum_multiplier_.Reshape(1, 1,
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_data();
  this->device_->set(sum_multiplier_.count(), Dtype(1), multiplier_data);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> device-abstraction
}

template <typename Dtype>
Dtype MVNLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->const_data();
  Dtype* top_data = (*top)[0]->mutable_data();
=======
<<<<<<< HEAD
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
  if ( this->layer_param_.mvn_param().across_channels() ) {
    sum_multiplier_.Reshape(1, bottom[0]->channels(), bottom[0]->height(),
                            bottom[0]->width());
  } else {
    sum_multiplier_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  }
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  eps_ = this->layer_param_.mvn_param().eps();
<<<<<<< HEAD
}

template <typename Dtype>
void MVNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
<<<<<<< HEAD
=======
  sum_multiplier_.Reshape(1, 1,
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_data();
  this->device_->set(sum_multiplier_.count(), Dtype(1), multiplier_data);
}

template <typename Dtype>
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> device-abstraction
=======
=======
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
  sum_multiplier_.Reshape(1, 1,
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_data();
  this->device_->set(sum_multiplier_.count(), Dtype(1), multiplier_data);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
}

template <typename Dtype>
>>>>>>> BVLC/device-abstraction
=======
}

template <typename Dtype>
>>>>>>> BVLC/device-abstraction
=======
}

template <typename Dtype>
>>>>>>> BVLC/device-abstraction
Dtype MVNLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->const_data();
  Dtype* top_data = (*top)[0]->mutable_data();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
  if ( this->layer_param_.mvn_param().across_channels() ) {
    sum_multiplier_.Reshape(1, bottom[0]->channels(), bottom[0]->height(),
                            bottom[0]->width());
  } else {
    sum_multiplier_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  }
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  eps_ = this->layer_param_.mvn_param().eps();
}

template <typename Dtype>
=======
}

template <typename Dtype>
>>>>>>> device-abstraction
void MVNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
  sum_multiplier_.Reshape(1, 1,
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_data();
  this->device_->set(sum_multiplier_.count(), Dtype(1), multiplier_data);
}

template <typename Dtype>
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
  sum_multiplier_.Reshape(1, 1,
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_data();
  this->device_->set(sum_multiplier_.count(), Dtype(1), multiplier_data);
}

template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
  sum_multiplier_.Reshape(1, 1,
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_data();
  this->device_->set(sum_multiplier_.count(), Dtype(1), multiplier_data);
}

template <typename Dtype>
>>>>>>> BVLC/device-abstraction
<<<<<<< HEAD
<<<<<<< HEAD
=======
}

template <typename Dtype>
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
}

template <typename Dtype>
>>>>>>> BVLC/device-abstraction
Dtype MVNLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->const_data();
  Dtype* top_data = (*top)[0]->mutable_data();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/master
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/master
>>>>>>> device-abstraction
  int num;
  if (this->layer_param_.mvn_param().across_channels()) {
    num = bottom[0]->num();
  } else {
    num = bottom[0]->num() * bottom[0]->channels();
  }

  int dim = bottom[0]->count() / num;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp

<<<<<<< HEAD
<<<<<<< HEAD
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  if (this->layer_param_.mvn_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_powx(bottom[0]->count(), top_data, Dtype(2),
        temp_.mutable_cpu_data());  // (X-EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
        sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E((X-EX)^2)
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge

=======
=======
>>>>>>> pod-caffe-pod.hpp-merge

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  if (this->layer_param_.mvn_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_powx(bottom[0]->count(), top_data, Dtype(2),
        temp_.mutable_cpu_data());  // (X-EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
        sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E((X-EX)^2)
>>>>>>> pod/device/blob.hpp
=======

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  if (this->layer_param_.mvn_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_powx(bottom[0]->count(), top_data, Dtype(2),
        temp_.mutable_cpu_data());  // (X-EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
        sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E((X-EX)^2)
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======

<<<<<<< HEAD
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======

<<<<<<< HEAD
>>>>>>> device-abstraction
  if (this->layer_param_.mvn_param().normalize_variance()) {
    // put the squares of bottom into temp_
    this->device_->powx(bottom[0]->count(), bottom_data, Dtype(2),
        temp_.mutable_data());

    // computes variance using var(X) = E(X^2) - (EX)^2
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
        sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, temp_.const_data(),
        sum_multiplier_.const_data(), 0.,
        variance_.mutable_data());  // E(X^2)
    this->device_->powx(mean_.count(), mean_.const_data(), Dtype(2),
        temp_.mutable_data());  // (EX)^2
    this->device_->sub(mean_.count(), variance_.const_data(),
        temp_.const_data(), variance_.mutable_data());  // variance
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp

    // do mean and variance normalization
    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
        mean_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======

=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> BVLC/master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> BVLC/master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> caffe
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
>>>>>>> pod/device/blob.hpp
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX
<<<<<<< HEAD
=======

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> BVLC/master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> caffe
  if (this->layer_param_.mvn_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_powx(bottom[0]->count(), top_data, Dtype(2),
        temp_.mutable_cpu_data());  // (X-EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
        sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E((X-EX)^2)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master

    // normalize variance
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());

<<<<<<< HEAD
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
=======
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
>>>>>>> BVLC/master

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.const_data(), sum_multiplier_.const_data(), 0.,
          temp_.mutable_data());

<<<<<<< HEAD
    this->device_->div(temp_.count(), top_data, temp_.const_data(), top_data);
  } else {
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> device-abstraction

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> device-abstraction
        mean_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
>>>>>>> pod/device/blob.hpp

=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> BVLC/master
=======
=======
>>>>>>> pod/device/blob.hpp
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> BVLC/master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> BVLC/master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> caffe
=======
>>>>>>> device-abstraction
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

<<<<<<< HEAD
>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> BVLC/master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> caffe
=======
>>>>>>> device-abstraction
  if (this->layer_param_.mvn_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_powx(bottom[0]->count(), top_data, Dtype(2),
        temp_.mutable_cpu_data());  // (X-EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
        sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E((X-EX)^2)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
        mean_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/master
>>>>>>> device-abstraction

    // normalize variance
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> device-abstraction
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
=======
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
>>>>>>> BVLC/master
<<<<<<< HEAD
=======
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> device-abstraction

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.const_data(), sum_multiplier_.const_data(), 0.,
          temp_.mutable_data());
<<<<<<< HEAD
<<<<<<< HEAD

<<<<<<< HEAD
    this->device_->div(temp_.count(), top_data, temp_.const_data(), top_data);
  } else {
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX
>>>>>>> pod-caffe-pod.hpp-merge

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
<<<<<<< HEAD
        mean_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction

    // normalize variance
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());

<<<<<<< HEAD
<<<<<<< HEAD
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
=======
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
>>>>>>> BVLC/device-abstraction
=======
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
>>>>>>> BVLC/device-abstraction

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.const_data(), sum_multiplier_.const_data(), 0.,
          temp_.mutable_data());

<<<<<<< HEAD
<<<<<<< HEAD
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
=======
=======
>>>>>>> BVLC/device-abstraction
=======

>>>>>>> BVLC/device-abstraction
=======

<<<<<<< HEAD
>>>>>>> device-abstraction
    this->device_->div(temp_.count(), top_data, temp_.const_data(), top_data);
  } else {
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());
<<<<<<< HEAD

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
=======
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
>>>>>>> BVLC/master
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
        mean_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
>>>>>>> BVLC/device-abstraction

    // normalize variance
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());

<<<<<<< HEAD
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
=======
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
>>>>>>> BVLC/device-abstraction

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.const_data(), sum_multiplier_.const_data(), 0.,
          temp_.mutable_data());

<<<<<<< HEAD
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
    this->device_->div(temp_.count(), top_data, temp_.const_data(), top_data);
  } else {
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());
=======
  }
>>>>>>> BVLC/device-abstraction

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
>>>>>>> BVLC/device-abstraction
  }
=======

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
=======
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
>>>>>>> BVLC/master
  }
>>>>>>> device-abstraction
}

template <typename Dtype>
void MVNLayer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> device-abstraction
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->const_diff();
  const Dtype* top_data = top[0]->const_data();
  const Dtype* bottom_data = (*bottom)[0]->const_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_diff();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> device-abstraction

  int num;
  if (this->layer_param_.mvn_param().across_channels()) {
    num = (*bottom)[0]->num();
  } else {
    num = (*bottom)[0]->num() * (*bottom)[0]->channels();
  }
=======
<<<<<<< HEAD
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
<<<<<<< HEAD
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> device-abstraction

=======
>>>>>>> BVLC/device-abstraction
  int num;
<<<<<<< HEAD
  if (this->layer_param_.mvn_param().across_channels()) {
<<<<<<< HEAD
<<<<<<< HEAD
    num = bottom[0]->num();
  } else {
    num = bottom[0]->num() * bottom[0]->channels();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

<<<<<<< HEAD
>>>>>>> BVLC/master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX
=======
  int num;
=======
>>>>>>> pod/device/blob.hpp
=======
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->const_diff();
  const Dtype* top_data = top[0]->const_data();
  const Dtype* bottom_data = (*bottom)[0]->const_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_diff();

  int num;
<<<<<<< HEAD
=======
=======
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->const_diff();
  const Dtype* top_data = top[0]->const_data();
  const Dtype* bottom_data = (*bottom)[0]->const_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_diff();

  int num;
>>>>>>> pod/device/blob.hpp
>>>>>>> BVLC/device-abstraction
  if (this->layer_param_.mvn_param().across_channels()) {
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
    num = (*bottom)[0]->num();
  } else {
    num = (*bottom)[0]->num() * (*bottom)[0]->channels();
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction

  int dim = bottom[0]->count() / num;
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/device-abstraction

  int dim = bottom[0]->count() / num;
=======
<<<<<<< HEAD
=======
=======
  }
>>>>>>> BVLC/device-abstraction

  int dim = bottom[0]->count() / num;
  Dtype eps = 1e-10;
>>>>>>> origin/BVLC/parallel
=======
=======
>>>>>>> BVLC/device-abstraction

<<<<<<< HEAD
  int dim = bottom[0]->count() / num;
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp

  int dim = bottom[0]->count() / num;
  Dtype eps = 1e-10;
>>>>>>> origin/BVLC/parallel
=======
=======
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();
>>>>>>> BVLC/master

  int dim = bottom[0]->count() / num;
>>>>>>> device-abstraction

  int dim = bottom[0]->count() / num;
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp

>>>>>>> caffe
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX
=======
  if (this->layer_param_.mvn_param().normalize_variance()) {
    this->device_->mul(temp_.count(), top_data, top_diff, bottom_diff);
    this->device_->gemv(CblasNoTrans, num, dim, 1., bottom_diff,
          sum_multiplier_.const_data(), 0., mean_.mutable_data());
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          mean_.const_data(), sum_multiplier_.const_data(), 0.,
          bottom_diff);
    this->device_->mul(temp_.count(), top_data, bottom_diff, bottom_diff);

    this->device_->gemv(CblasNoTrans, num, dim, 1., top_diff,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
            mean_.const_data(), sum_multiplier_.const_data(), 1.,
            bottom_diff);

    this->device_->axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1. / dim),
        bottom_diff);
>>>>>>> BVLC/device-abstraction

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

<<<<<<< HEAD
>>>>>>> BVLC/master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX
=======
    // put the squares of bottom into temp_
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    caffe_powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
        temp_.mutable_cpu_data());
=======
=======
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
    this->device_->powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_data());
>>>>>>> pod/device/blob.hpp

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> caffe
  if (this->layer_param_.mvn_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_powx(bottom[0]->count(), top_data, Dtype(2),
        temp_.mutable_cpu_data());  // (X-EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
        sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E((X-EX)^2)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
    this->device_->powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_data());

=======
    this->device_->powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_data());

>>>>>>> BVLC/device-abstraction
=======
    this->device_->powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_data());

>>>>>>> device-abstraction
    // computes variance using var(X) = E(X^2) - (EX)^2
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
        sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, temp_.const_data(),
        sum_multiplier_.const_data(), 0.,
        variance_.mutable_data());  // E(X^2)
    this->device_->powx(mean_.count(), mean_.const_data(), Dtype(2),
        temp_.mutable_data());  // (EX)^2
    this->device_->sub(mean_.count(), variance_.const_data(),
        temp_.const_data(), variance_.mutable_data());  // variance
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> device-abstraction

    // normalize variance
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());
<<<<<<< HEAD
<<<<<<< HEAD

<<<<<<< HEAD
<<<<<<< HEAD
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
=======
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
>>>>>>> BVLC/master

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
<<<<<<< HEAD
          variance_.const_data(), sum_multiplier_.const_data(), 0.,
          temp_.mutable_data());
=======
        variance_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
    caffe_powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
        temp_.mutable_cpu_data());
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp

<<<<<<< HEAD
    this->device_->div(temp_.count(), top_data, temp_.const_data(), top_data);
  } else {
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, top_diff,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
    caffe_add(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
<<<<<<< HEAD
=======
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
>>>>>>> BVLC/device-abstraction
=======
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  }
}
>>>>>>> pod/device/blob.hpp
=======
    this->device_->powx(bottom[0]->count(), bottom_data, Dtype(2),
        temp_.mutable_data());

    // computes variance using var(X) = E(X^2) - (EX)^2
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
        sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, temp_.const_data(),
        sum_multiplier_.const_data(), 0.,
        variance_.mutable_data());  // E(X^2)
    this->device_->powx(mean_.count(), mean_.const_data(), Dtype(2),
        temp_.mutable_data());  // (EX)^2
    this->device_->sub(mean_.count(), variance_.const_data(),
        temp_.const_data(), variance_.mutable_data());  // variance
>>>>>>> BVLC/device-abstraction

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
<<<<<<< HEAD
<<<<<<< HEAD
        mean_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());
<<<<<<< HEAD

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
>>>>>>> pod/device/blob.hpp

    // normalize variance
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======

    this->device_->add_scalar(variance_.count(), eps, variance_.mutable_data());

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());
=======
>>>>>>> device-abstraction
    caffe_powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
        temp_.mutable_cpu_data());
>>>>>>> BVLC/master
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
=======
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, top_diff,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
    caffe_add(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
>>>>>>> BVLC/master
=======
<<<<<<< HEAD
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
=======
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
>>>>>>> BVLC/device-abstraction
>>>>>>> pod/device/blob.hpp
=======
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
    this->device_->add_scalar(variance_.count(), eps, variance_.mutable_data());

=======

    this->device_->add_scalar(variance_.count(), eps, variance_.mutable_data());

>>>>>>> BVLC/device-abstraction
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());

    this->device_->div(temp_.count(), bottom_diff, temp_.const_data(),
        bottom_diff);
  } else {
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======

    this->device_->div(temp_.count(), bottom_diff, temp_.const_data(),
        bottom_diff);
  } else {
<<<<<<< HEAD
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
=======
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, top_diff,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
    caffe_add(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
>>>>>>> BVLC/master
>>>>>>> device-abstraction
  }
}
>>>>>>> pod/device/blob.hpp

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.const_data(), sum_multiplier_.const_data(), 0.,
          temp_.mutable_data());

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
=======
=======
>>>>>>> BVLC/device-abstraction
>>>>>>> pod/device/blob.hpp
    this->device_->div(temp_.count(), top_data, temp_.const_data(), top_data);
  } else {
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> device-abstraction

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> device-abstraction
        mean_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction

    // normalize variance
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());

<<<<<<< HEAD
<<<<<<< HEAD
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());

<<<<<<< HEAD
    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
=======

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
>>>>>>> pod-caffe-pod.hpp-merge
=======
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
>>>>>>> BVLC/master
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
        mean_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
>>>>>>> BVLC/device-abstraction

<<<<<<< HEAD
    // normalize variance
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());

<<<<<<< HEAD
<<<<<<< HEAD
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
=======
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
>>>>>>> BVLC/device-abstraction

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.const_data(), sum_multiplier_.const_data(), 0.,
          temp_.mutable_data());

<<<<<<< HEAD
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
  }
}

template <typename Dtype>
void MVNLayer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->const_diff();
  const Dtype* top_data = top[0]->const_data();
  const Dtype* bottom_data = (*bottom)[0]->const_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_diff();

  int num;
  if (this->layer_param_.mvn_param().across_channels()) {
    num = (*bottom)[0]->num();
  } else {
    num = (*bottom)[0]->num() * (*bottom)[0]->channels();
  }
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
>>>>>>> BVLC/device-abstraction
=======
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
>>>>>>> BVLC/device-abstraction
>>>>>>> pod/device/blob.hpp

=======
>>>>>>> device-abstraction
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX
=======
<<<<<<< HEAD
<<<<<<< HEAD
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
=======
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
    this->device_->div(temp_.count(), top_data, temp_.const_data(), top_data);
  } else {
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp

>>>>>>> BVLC/master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

<<<<<<< HEAD
>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> caffe
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
=======
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
>>>>>>> BVLC/master
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> BVLC/device-abstraction

    // normalize variance
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());

<<<<<<< HEAD
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
=======
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
>>>>>>> BVLC/device-abstraction

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.const_data(), sum_multiplier_.const_data(), 0.,
          temp_.mutable_data());

<<<<<<< HEAD
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  }
}

template <typename Dtype>
void MVNLayer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->const_diff();
  const Dtype* top_data = top[0]->const_data();
  const Dtype* bottom_data = (*bottom)[0]->const_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_diff();

  int num;
  if (this->layer_param_.mvn_param().across_channels()) {
    num = (*bottom)[0]->num();
  } else {
    num = (*bottom)[0]->num() * (*bottom)[0]->channels();
  }
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> BVLC/master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> caffe
  if (this->layer_param_.mvn_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_powx(bottom[0]->count(), top_data, Dtype(2),
        temp_.mutable_cpu_data());  // (X-EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
        sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E((X-EX)^2)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master

    // normalize variance
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());

<<<<<<< HEAD
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
=======
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
>>>>>>> BVLC/master

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.const_data(), sum_multiplier_.const_data(), 0.,
          temp_.mutable_data());

<<<<<<< HEAD
    this->device_->div(temp_.count(), top_data, temp_.const_data(), top_data);
  } else {
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
<<<<<<< HEAD
        mean_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());
=======
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());
>>>>>>> pod-caffe-pod.hpp-merge

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
=======
<<<<<<< HEAD
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> BVLC/master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> BVLC/master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> caffe
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> BVLC/master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> master
=======
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

>>>>>>> caffe
=======
>>>>>>> device-abstraction
  if (this->layer_param_.mvn_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_powx(bottom[0]->count(), top_data, Dtype(2),
        temp_.mutable_cpu_data());  // (X-EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
        sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E((X-EX)^2)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master

    // normalize variance
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());

<<<<<<< HEAD
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
=======
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
>>>>>>> BVLC/master

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.const_data(), sum_multiplier_.const_data(), 0.,
          temp_.mutable_data());

<<<<<<< HEAD
    this->device_->div(temp_.count(), top_data, temp_.const_data(), top_data);
  } else {
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX
>>>>>>> pod-caffe-pod.hpp-merge
=======

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
>>>>>>> pod/caffe-merge
=======

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
>>>>>>> pod/caffe-merge
=======

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
=======

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
>>>>>>> BVLC/device-abstraction
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
>>>>>>> pod/caffe-merge
=======
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
>>>>>>> BVLC/master
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe

    // normalize variance
    caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
          variance_.mutable_cpu_data());

    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
<<<<<<< HEAD
        mean_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
>>>>>>> BVLC/device-abstraction
=======
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
=======
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge

    // normalize variance
=======
>>>>>>> BVLC/master

    // normalize variance
>>>>>>> device-abstraction
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
=======
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
>>>>>>> BVLC/device-abstraction
=======
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
>>>>>>> pod-caffe-pod.hpp-merge
=======
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
=======
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
>>>>>>> BVLC/master
>>>>>>> device-abstraction

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.const_data(), sum_multiplier_.const_data(), 0.,
          temp_.mutable_data());

<<<<<<< HEAD
<<<<<<< HEAD
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> device-abstraction
    this->device_->div(temp_.count(), top_data, temp_.const_data(), top_data);
  } else {
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());
<<<<<<< HEAD
=======
    this->device_->div(temp_.count(), top_data, temp_.const_data(), top_data);
  } else {
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
=======
>>>>>>> BVLC/device-abstraction
  }
>>>>>>> BVLC/device-abstraction

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
  }
=======

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
=======
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
>>>>>>> BVLC/master
  }
>>>>>>> device-abstraction
=======
  }
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
  }
>>>>>>> pod/caffe-merge
=======
  }
>>>>>>> pod-caffe-pod.hpp-merge
}

template <typename Dtype>
void MVNLayer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->const_diff();
  const Dtype* top_data = top[0]->const_data();
  const Dtype* bottom_data = (*bottom)[0]->const_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_diff();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge

  int num;
  if (this->layer_param_.mvn_param().across_channels()) {
    num = (*bottom)[0]->num();
  } else {
    num = (*bottom)[0]->num() * (*bottom)[0]->channels();
  }
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->const_diff();
  const Dtype* top_data = top[0]->const_data();
  const Dtype* bottom_data = (*bottom)[0]->const_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_diff();

  int num;
=======
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->const_diff();
  const Dtype* top_data = top[0]->const_data();
  const Dtype* bottom_data = (*bottom)[0]->const_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_diff();

  int num;
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
=======
=======
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->const_diff();
  const Dtype* top_data = top[0]->const_data();
  const Dtype* bottom_data = (*bottom)[0]->const_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_diff();
>>>>>>> pod/device/blob.hpp

  int num;
>>>>>>> BVLC/device-abstraction
=======

  int num;
>>>>>>> BVLC/device-abstraction
  if (this->layer_param_.mvn_param().across_channels()) {
    num = (*bottom)[0]->num();
  } else {
    num = (*bottom)[0]->num() * (*bottom)[0]->channels();
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction

  int dim = bottom[0]->count() / num;
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp

  int dim = bottom[0]->count() / num;
=======
=======
>>>>>>> BVLC/device-abstraction
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction

  int dim = bottom[0]->count() / num;
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master

  int dim = bottom[0]->count() / num;
=======

  int dim = bottom[0]->count() / num;
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======

  int dim = bottom[0]->count() / num;
=======

<<<<<<< HEAD
  int dim = bottom[0]->count() / num;
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
=======
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
>>>>>>> BVLC/device-abstraction
=======
    this->device_->add_scalar(variance_.count(), eps,
        variance_.mutable_data());
>>>>>>> BVLC/device-abstraction
>>>>>>> pod/device/blob.hpp

  int dim = bottom[0]->count() / num;
=======

  int dim = bottom[0]->count() / num;
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master

  int dim = bottom[0]->count() / num;
=======
=======
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
=======
=======
>>>>>>> BVLC/device-abstraction
    this->device_->div(temp_.count(), top_data, temp_.const_data(), top_data);
  } else {
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX

    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
            mean_.const_data(), sum_multiplier_.const_data(), 0.,
            temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
=======
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
>>>>>>> BVLC/master
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe

    // normalize variance
    caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
          variance_.mutable_cpu_data());

    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
          temp_.mutable_cpu_data());

    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  }
}
>>>>>>> pod/device/blob.hpp

  int dim = bottom[0]->count() / num;
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
  Dtype eps = 1e-10;
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->const_diff();
  const Dtype* top_data = top[0]->const_data();
  const Dtype* bottom_data = (*bottom)[0]->const_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_diff();

  int num;
  if (this->layer_param_.mvn_param().across_channels()) {
    num = (*bottom)[0]->num();
  } else {
    num = (*bottom)[0]->num() * (*bottom)[0]->channels();
  }
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======

<<<<<<< HEAD
  int dim = bottom[0]->count() / num;
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/master

  int dim = bottom[0]->count() / num;
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
  int num;
=======
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->const_diff();
  const Dtype* top_data = top[0]->const_data();
  const Dtype* bottom_data = (*bottom)[0]->const_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_diff();

  int num;
>>>>>>> BVLC/device-abstraction
  if (this->layer_param_.mvn_param().across_channels()) {
    num = (*bottom)[0]->num();
  } else {
    num = (*bottom)[0]->num() * (*bottom)[0]->channels();
  }
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction

  int dim = bottom[0]->count() / num;
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master

  int dim = bottom[0]->count() / num;
=======

  int dim = bottom[0]->count() / num;
  Dtype eps = 1e-10;
>>>>>>> origin/BVLC/parallel
=======

  int dim = bottom[0]->count() / num;
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge

  if (this->layer_param_.mvn_param().normalize_variance()) {
    this->device_->mul(temp_.count(), top_data, top_diff, bottom_diff);
    this->device_->gemv(CblasNoTrans, num, dim, 1., bottom_diff,
          sum_multiplier_.const_data(), 0., mean_.mutable_data());
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          mean_.const_data(), sum_multiplier_.const_data(), 0.,
          bottom_diff);
    this->device_->mul(temp_.count(), top_data, bottom_diff, bottom_diff);

    this->device_->gemv(CblasNoTrans, num, dim, 1., top_diff,
            sum_multiplier_.const_data(), 0., mean_.mutable_data());
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
            mean_.const_data(), sum_multiplier_.const_data(), 1.,
            bottom_diff);

    this->device_->axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1. / dim),
        bottom_diff);

    // put the squares of bottom into temp_
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
    caffe_powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
        temp_.mutable_cpu_data());
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> BVLC/device-abstraction
    this->device_->powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_data());

    // computes variance using var(X) = E(X^2) - (EX)^2
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
        sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, temp_.const_data(),
        sum_multiplier_.const_data(), 0.,
        variance_.mutable_data());  // E(X^2)
    this->device_->powx(mean_.count(), mean_.const_data(), Dtype(2),
        temp_.mutable_data());  // (EX)^2
    this->device_->sub(mean_.count(), variance_.const_data(),
        temp_.const_data(), variance_.mutable_data());  // variance

    // normalize variance
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());

    this->device_->add_scalar(variance_.count(), eps, variance_.mutable_data());

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
    caffe_powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
        temp_.mutable_cpu_data());
>>>>>>> BVLC/master
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> pod/device/blob.hpp
    caffe_powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
        temp_.mutable_cpu_data());
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
=======
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
    this->device_->powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_data());

    // computes variance using var(X) = E(X^2) - (EX)^2
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
        sum_multiplier_.const_data(), 0., mean_.mutable_data());  // EX
    this->device_->gemv(CblasNoTrans, num, dim, 1. / dim, temp_.const_data(),
        sum_multiplier_.const_data(), 0.,
        variance_.mutable_data());  // E(X^2)
    this->device_->powx(mean_.count(), mean_.const_data(), Dtype(2),
        temp_.mutable_data());  // (EX)^2
    this->device_->sub(mean_.count(), variance_.const_data(),
        temp_.const_data(), variance_.mutable_data());  // variance

    // normalize variance
    this->device_->powx(variance_.count(), variance_.const_data(), Dtype(0.5),
          variance_.mutable_data());

    this->device_->add_scalar(variance_.count(), eps, variance_.mutable_data());

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
    caffe_powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
        temp_.mutable_cpu_data());
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge

    this->device_->div(temp_.count(), bottom_diff, temp_.const_data(),
        bottom_diff);
  } else {
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, top_diff,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
    caffe_add(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
>>>>>>> BVLC/master
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
>>>>>>> BVLC/device-abstraction
=======
    this->device_->copy(temp_.count(), top_diff, bottom_diff);
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  }
}


INSTANTIATE_CLASS(MVNLayer);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
INSTANTIATE_CLASS(MVNLayer);
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
REGISTER_LAYER_CLASS(MVN);

=======
REGISTER_LAYER_CLASS(MVN, MVNLayer);
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
INSTANTIATE_CLASS(MVNLayer);
>>>>>>> device-abstraction
REGISTER_LAYER_CLASS(MVN);

>>>>>>> caffe
}  // namespace caffe
