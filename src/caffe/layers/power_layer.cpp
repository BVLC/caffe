#include <vector>

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
=======
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
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
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> caffe
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge

namespace caffe {

template <typename Dtype>
void PowerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  power_ = this->layer_param_.power_param().power();
  scale_ = this->layer_param_.power_param().scale();
  shift_ = this->layer_param_.power_param().shift();
  diff_scale_ = power_  * scale_;
}

// Compute y = (shift + scale * x)^power
template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
=======
Dtype PowerLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype* top_data = (*top)[0]->mutable_data();
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
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
Dtype PowerLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype* top_data = (*top)[0]->mutable_data();
=======
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
>>>>>>> BVLC/master
=======
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
>>>>>>> BVLC/master
=======
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
>>>>>>> BVLC/master
=======
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
>>>>>>> master
=======
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
>>>>>>> caffe
=======
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
>>>>>>> master
=======
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
>>>>>>> master
=======
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
>>>>>>> BVLC/master
=======
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
>>>>>>> master
=======
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
>>>>>>> master
=======
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
>>>>>>> origin/BVLC/parallel
=======
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
  const int count = bottom[0]->count();
  // Special case where we can ignore the input: scale or power is 0.
  if (diff_scale_ == Dtype(0)) {
    Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    caffe_set(count, value, top_data);
    return;
=======
    this->device_->set(count, value, top_data);
    return Dtype(0);
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
    this->device_->set(count, value, top_data);
    return Dtype(0);
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
    this->device_->set(count, value, top_data);
    return Dtype(0);
>>>>>>> pod-caffe-pod.hpp-merge
=======
    caffe_set(count, value, top_data);
    return;
>>>>>>> BVLC/master
=======
    caffe_set(count, value, top_data);
    return;
>>>>>>> BVLC/master
=======
    caffe_set(count, value, top_data);
    return;
>>>>>>> BVLC/master
=======
    caffe_set(count, value, top_data);
    return;
>>>>>>> master
=======
    caffe_set(count, value, top_data);
    return;
>>>>>>> caffe
=======
    caffe_set(count, value, top_data);
    return;
>>>>>>> master
=======
    caffe_set(count, value, top_data);
    return;
>>>>>>> master
=======
    caffe_set(count, value, top_data);
    return;
>>>>>>> BVLC/master
=======
    caffe_set(count, value, top_data);
    return;
>>>>>>> master
=======
    caffe_set(count, value, top_data);
    return;
>>>>>>> master
=======
    caffe_set(count, value, top_data);
    return;
>>>>>>> origin/BVLC/parallel
=======
    caffe_set(count, value, top_data);
    return;
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
  }
  const Dtype* bottom_data = bottom[0]->const_data();
  this->device_->copy(count, bottom_data, top_data);
  if (scale_ != Dtype(1)) {
    this->device_->scal(count, scale_, top_data);
  }
  if (shift_ != Dtype(0)) {
    this->device_->add_scalar(count, shift_, top_data);
  }
  if (power_ != Dtype(1)) {
    this->device_->powx(count, top_data, power_, top_data);
  }
}

template <typename Dtype>
void PowerLayer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
=======
    Dtype* bottom_diff = (*bottom)[0]->mutable_diff();
    const int count = (*bottom)[0]->count();
    const Dtype* top_diff = top[0]->const_diff();
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
    Dtype* bottom_diff = (*bottom)[0]->mutable_diff();
    const int count = (*bottom)[0]->count();
    const Dtype* top_diff = top[0]->const_diff();
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
    Dtype* bottom_diff = (*bottom)[0]->mutable_diff();
    const int count = (*bottom)[0]->count();
    const Dtype* top_diff = top[0]->const_diff();
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
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
>>>>>>> BVLC/master
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
    if (diff_scale_ == Dtype(0) || power_ == Dtype(1)) {
      this->device_->set(count, diff_scale_, bottom_diff);
    } else {
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
      const Dtype* bottom_data = bottom[0]->cpu_data();
=======
      const Dtype* bottom_data = (*bottom)[0]->const_data();
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
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
      const Dtype* bottom_data = (*bottom)[0]->const_data();
=======
      const Dtype* bottom_data = bottom[0]->cpu_data();
>>>>>>> BVLC/master
=======
      const Dtype* bottom_data = bottom[0]->cpu_data();
>>>>>>> BVLC/master
=======
      const Dtype* bottom_data = bottom[0]->cpu_data();
>>>>>>> BVLC/master
=======
      const Dtype* bottom_data = bottom[0]->cpu_data();
>>>>>>> master
=======
      const Dtype* bottom_data = bottom[0]->cpu_data();
>>>>>>> caffe
=======
      const Dtype* bottom_data = bottom[0]->cpu_data();
>>>>>>> master
=======
      const Dtype* bottom_data = bottom[0]->cpu_data();
>>>>>>> master
=======
      const Dtype* bottom_data = bottom[0]->cpu_data();
>>>>>>> BVLC/master
=======
      const Dtype* bottom_data = bottom[0]->cpu_data();
>>>>>>> master
=======
      const Dtype* bottom_data = bottom[0]->cpu_data();
>>>>>>> master
=======
      const Dtype* bottom_data = bottom[0]->cpu_data();
>>>>>>> origin/BVLC/parallel
=======
      const Dtype* bottom_data = bottom[0]->cpu_data();
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
      // Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
      //               = diff_scale * y / (shift + scale * x)
      if (power_ == Dtype(2)) {
        // Special case for y = (shift + scale * x)^2
        //     -> dy/dx = 2 * scale * (shift + scale * x)
        //              = diff_scale * shift + diff_scale * scale * x
        this->device_->axpby(count, diff_scale_ * scale_, bottom_data,
            Dtype(0), bottom_diff);
        if (shift_ != Dtype(0)) {
          this->device_->add_scalar(count, diff_scale_ * shift_, bottom_diff);
        }
      } else if (shift_ == Dtype(0)) {
        // Special case for y = (scale * x)^power
        //     -> dy/dx = scale * power * (scale * x)^(power - 1)
        //              = scale * power * (scale * x)^power * (scale * x)^(-1)
        //              = power * y / x
        const Dtype* top_data = top[0]->const_data();
        this->device_->div(count, top_data, bottom_data, bottom_diff);
        this->device_->scal(count, power_, bottom_diff);
      } else {
        this->device_->copy(count, bottom_data, bottom_diff);
        if (scale_ != Dtype(1)) {
          this->device_->scal(count, scale_, bottom_diff);
        }
        if (shift_ != Dtype(0)) {
          this->device_->add_scalar(count, shift_, bottom_diff);
        }
        const Dtype* top_data = top[0]->const_data();
        this->device_->div(count, top_data, bottom_diff, bottom_diff);
        if (diff_scale_ != Dtype(1)) {
          this->device_->scal(count, diff_scale_, bottom_diff);
        }
      }
    }
    if (diff_scale_ != Dtype(0)) {
      this->device_->mul(count, top_diff, bottom_diff, bottom_diff);
    }
  }
}

INSTANTIATE_CLASS(PowerLayer);
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
REGISTER_LAYER_CLASS(Power);

=======
REGISTER_LAYER_CLASS(POWER, PowerLayer);
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
REGISTER_LAYER_CLASS(Power);

>>>>>>> caffe
}  // namespace caffe
