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
=======
>>>>>>> pod/device/blob.hpp
#include "caffe/loss_layers.hpp"
=======
#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
>>>>>>> BVLC/master
#include "caffe/util/math_functions.hpp"
=======
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
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
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
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
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> device-abstraction
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
<<<<<<< HEAD
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

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
}

template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
<<<<<<< HEAD
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
=======
=======
>>>>>>> BVLC/device-abstraction
Dtype SigmoidCrossEntropyLossLayer<Dtype>::Forward(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
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
Dtype SigmoidCrossEntropyLossLayer<Dtype>::Forward(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
=======
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
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
Dtype SigmoidCrossEntropyLossLayer<Dtype>::Forward(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
}

template <typename Dtype>
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
=======
>>>>>>> BVLC/device-abstraction
Dtype SigmoidCrossEntropyLossLayer<Dtype>::Forward(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
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
}

template <typename Dtype>
>>>>>>> caffe
=======
Dtype SigmoidCrossEntropyLossLayer<Dtype>::Forward(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
=======
>>>>>>> device-abstraction
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
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
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
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
>>>>>>> BVLC/master
>>>>>>> device-abstraction
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> device-abstraction
    const int count = (*bottom)[0]->count();
    const int num = (*bottom)[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->const_data();
    const Dtype* target = (*bottom)[1]->const_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_diff();
    this->device_->sub(count, sigmoid_output_data, target, bottom_diff);
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> device-abstraction
    // Scale down gradient
    this->device_->scal(count, Dtype(1) / num, bottom_diff);
  }
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
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidCrossEntropyLossLayer, Backward);
#endif

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
    const int count = (*bottom)[0]->count();
    const int num = (*bottom)[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->const_data();
    const Dtype* target = (*bottom)[1]->const_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_diff();
    this->device_->sub(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    this->device_->scal(count, Dtype(1) / num, bottom_diff);
  }
}
>>>>>>> pod/device/blob.hpp
=======
    const int count = (*bottom)[0]->count();
    const int num = (*bottom)[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->const_data();
    const Dtype* target = (*bottom)[1]->const_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_diff();
    this->device_->sub(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    this->device_->scal(count, Dtype(1) / num, bottom_diff);
  }
}

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
>>>>>>> BVLC/device-abstraction

>>>>>>> BVLC/device-abstraction
INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> BVLC/master
=======
    // Scale down gradient
    this->device_->scal(count, Dtype(1) / num, bottom_diff);
  }
}

>>>>>>> BVLC/device-abstraction
INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
<<<<<<< HEAD
<<<<<<< HEAD
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

=======
REGISTER_LAYER_CLASS(SIGMOID_CROSS_ENTROPY_LOSS, SigmoidCrossEntropyLossLayer);
>>>>>>> origin/BVLC/parallel
=======
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
INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
>>>>>>> device-abstraction
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

>>>>>>> caffe
}  // namespace caffe
