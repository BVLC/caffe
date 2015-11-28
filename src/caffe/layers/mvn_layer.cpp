#include <vector>

#include "caffe/common_layers.hpp"
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/util/math_functions.hpp"
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
  sum_multiplier_.Reshape(1, 1,
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_data();
  this->device_->set(sum_multiplier_.count(), Dtype(1), multiplier_data);
}

template <typename Dtype>
Dtype MVNLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->const_data();
  Dtype* top_data = (*top)[0]->mutable_data();
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
Dtype MVNLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->const_data();
  Dtype* top_data = (*top)[0]->mutable_data();
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
>>>>>>> pod-caffe-pod.hpp-merge
  int num;
  if (this->layer_param_.mvn_param().across_channels()) {
    num = bottom[0]->num();
  } else {
    num = bottom[0]->num() * bottom[0]->channels();
  }

  int dim = bottom[0]->count() / num;

<<<<<<< HEAD
<<<<<<< HEAD
  // subtract mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

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
=======

    // do mean and variance normalization
    // subtract mean
    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
        mean_.const_data(), sum_multiplier_.const_data(), 0.,
        temp_.mutable_data());

    this->device_->add(temp_.count(), bottom_data, temp_.const_data(),
        top_data);
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
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
>>>>>>> pod-caffe-pod.hpp-merge

    this->device_->gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.const_data(), sum_multiplier_.const_data(), 0.,
          temp_.mutable_data());

<<<<<<< HEAD
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
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
=======
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
    caffe_powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
        temp_.mutable_cpu_data());
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
=======
<<<<<<< HEAD
<<<<<<< HEAD
REGISTER_LAYER_CLASS(MVN);

=======
REGISTER_LAYER_CLASS(MVN, MVNLayer);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
REGISTER_LAYER_CLASS(MVN);

>>>>>>> caffe
}  // namespace caffe
