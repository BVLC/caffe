#include <algorithm>
#include <vector>

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
#include "caffe/loss_layers.hpp"
=======
#include "caffe/layers/hinge_loss_layer.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/layers/hinge_loss_layer.hpp"
>>>>>>> BVLC/master
#include "caffe/util/math_functions.hpp"
=======
=======
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
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"
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
=======
>>>>>>> BVLC/device-abstraction
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> caffe
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> device-abstraction
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
<<<<<<< HEAD
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
>>>>>>> pod/caffe-merge
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
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
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge

namespace caffe {

template <typename Dtype>
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
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
=======
=======
>>>>>>> BVLC/device-abstraction
Dtype HingeLossLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> BVLC/device-abstraction
Dtype HingeLossLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
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
Dtype HingeLossLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
Dtype HingeLossLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
Dtype HingeLossLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
=======
>>>>>>> pod/caffe-merge
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
>>>>>>> BVLC/master
=======
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
>>>>>>> BVLC/master
=======
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
>>>>>>> BVLC/master
=======
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
>>>>>>> master
=======
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
>>>>>>> caffe
=======
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
>>>>>>> master
=======
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
>>>>>>> master
=======
<<<<<<< HEAD
<<<<<<< HEAD
Dtype HingeLossLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
>>>>>>> device-abstraction
=======
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
>>>>>>> BVLC/master
<<<<<<< HEAD
=======
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
>>>>>>> BVLC/master
>>>>>>> pod/caffe-merge
=======
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
>>>>>>> master
=======
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
>>>>>>> master
=======
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
>>>>>>> origin/BVLC/parallel
=======
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
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
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  GetDevice<Dtype>(Caffe::CPU)->copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
  }
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_diff[i * dim + j] = std::max(
        Dtype(0), 1 + bottom_diff[i * dim + j]);
    }
  }
  Dtype* loss = top[0]->mutable_cpu_data();
  switch (this->layer_param_.hinge_loss_param().norm()) {
  case HingeLossParameter_Norm_L1:
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
<<<<<<< HEAD
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
    Dtype asum;
    GetDevice<Dtype>(Caffe::CPU)->asum(count, bottom_diff, &asum);
    return asum / num;
  case HingeLossParameter_Norm_L2:
    Dtype dot;
    GetDevice<Dtype>(Caffe::CPU)->dot(count, bottom_diff, bottom_diff, &dot);
    return dot / num;
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
    loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
    break;
  case HingeLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    break;
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
>>>>>>> origin/BVLC/parallel
    loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
    break;
  case HingeLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    break;
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
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
    loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
    break;
  case HingeLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    break;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
    Dtype asum;
    GetDevice<Dtype>(Caffe::CPU)->asum(count, bottom_diff, &asum);
    return asum / num;
  case HingeLossParameter_Norm_L2:
    Dtype dot;
    GetDevice<Dtype>(Caffe::CPU)->dot(count, bottom_diff, bottom_diff, &dot);
    return dot / num;
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
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
>>>>>>> origin/BVLC/parallel
=======
    Dtype asum;
    GetDevice<Dtype>(Caffe::CPU)->asum(count, bottom_diff, &asum);
    return asum / num;
  case HingeLossParameter_Norm_L2:
    Dtype dot;
    GetDevice<Dtype>(Caffe::CPU)->dot(count, bottom_diff, bottom_diff, &dot);
    return dot / num;
=======
>>>>>>> device-abstraction
    loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
    break;
  case HingeLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    break;
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
    loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
    break;
  case HingeLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    break;
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
    Dtype asum;
    GetDevice<Dtype>(Caffe::CPU)->asum(count, bottom_diff, &asum);
    return asum / num;
  case HingeLossParameter_Norm_L2:
    Dtype dot;
    GetDevice<Dtype>(Caffe::CPU)->dot(count, bottom_diff, bottom_diff, &dot);
    return dot / num;
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/master
>>>>>>> device-abstraction
=======
    loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
    break;
  case HingeLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    break;
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
    loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
    break;
  case HingeLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    break;
>>>>>>> caffe
>>>>>>> pod/caffe-merge
  default:
    LOG(FATAL) << "Unknown Norm";
  }
}

template <typename Dtype>
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
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
=======
=======
>>>>>>> BVLC/device-abstraction
void HingeLossLayer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
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
<<<<<<< HEAD
=======
>>>>>>> BVLC/device-abstraction
void HingeLossLayer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
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
void HingeLossLayer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
void HingeLossLayer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
void HingeLossLayer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
>>>>>>> BVLC/master
=======
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
>>>>>>> BVLC/master
=======
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
>>>>>>> BVLC/master
=======
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
>>>>>>> master
=======
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
>>>>>>> caffe
=======
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
>>>>>>> master
=======
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
>>>>>>> master
=======
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
>>>>>>> BVLC/master
=======
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
>>>>>>> master
=======
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
>>>>>>> master
=======
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
>>>>>>> origin/BVLC/parallel
=======
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
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
void HingeLossLayer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
=======
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
>>>>>>> BVLC/master
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    for (int i = 0; i < num; ++i) {
      bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
    }

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    switch (this->layer_param_.hinge_loss_param().norm()) {
    case HingeLossParameter_Norm_L1:
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
      caffe_cpu_sign(count, bottom_diff, bottom_diff);
      caffe_scal(count, loss_weight / num, bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      caffe_scal(count, loss_weight * 2 / num, bottom_diff);
=======
=======
>>>>>>> BVLC/device-abstraction
      GetDevice<Dtype>(Caffe::CPU)->sign(count, bottom_diff, bottom_diff);
      GetDevice<Dtype>(Caffe::CPU)->scal(count, Dtype(1. / num), bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      GetDevice<Dtype>(Caffe::CPU)->scal(count, Dtype(2. / num), bottom_diff);
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
      GetDevice<Dtype>(Caffe::CPU)->sign(count, bottom_diff, bottom_diff);
      GetDevice<Dtype>(Caffe::CPU)->scal(count, Dtype(1. / num), bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      GetDevice<Dtype>(Caffe::CPU)->scal(count, Dtype(2. / num), bottom_diff);
=======
      caffe_cpu_sign(count, bottom_diff, bottom_diff);
      caffe_scal(count, loss_weight / num, bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      caffe_scal(count, loss_weight * 2 / num, bottom_diff);
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
      GetDevice<Dtype>(Caffe::CPU)->sign(count, bottom_diff, bottom_diff);
      GetDevice<Dtype>(Caffe::CPU)->scal(count, Dtype(1. / num), bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      GetDevice<Dtype>(Caffe::CPU)->scal(count, Dtype(2. / num), bottom_diff);
>>>>>>> BVLC/device-abstraction
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
      GetDevice<Dtype>(Caffe::CPU)->sign(count, bottom_diff, bottom_diff);
      GetDevice<Dtype>(Caffe::CPU)->scal(count, Dtype(1. / num), bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      GetDevice<Dtype>(Caffe::CPU)->scal(count, Dtype(2. / num), bottom_diff);
=======
      caffe_cpu_sign(count, bottom_diff, bottom_diff);
      caffe_scal(count, loss_weight / num, bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      caffe_scal(count, loss_weight * 2 / num, bottom_diff);
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
>>>>>>> device-abstraction
=======
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
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
  }
}

INSTANTIATE_CLASS(HingeLossLayer);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
REGISTER_LAYER_CLASS(HingeLoss);

=======
REGISTER_LAYER_CLASS(HINGE_LOSS, HingeLossLayer);
>>>>>>> origin/BVLC/parallel
}  // namespace caffe
