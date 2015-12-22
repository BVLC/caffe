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
#include "caffe/loss_layers.hpp"
=======
#include "caffe/layers/contrastive_loss_layer.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/layers/contrastive_loss_layer.hpp"
>>>>>>> BVLC/master
#include "caffe/util/math_functions.hpp"
=======
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
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
=======
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
=======
>>>>>>> pod/caffe-merge
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
=======
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
=======
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
=======
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
=======
>>>>>>> pod/device/blob.hpp
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> origin/BVLC/parallel
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
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
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
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> device-abstraction

namespace caffe {

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      diff_.mutable_gpu_data());  // a_i-b_i
  caffe_gpu_powx(
      count,
      diff_.mutable_gpu_data(),  // a_i-b_i
      Dtype(2),
      diff_sq_.mutable_gpu_data());  // (a_i-b_i)^2
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_.gpu_data(),  // (a_i-b_i)^2
      summer_vec_.gpu_data(),
      Dtype(0.0),
      dist_sq_.mutable_gpu_data());  // \Sum (a_i-b_i)^2
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
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
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod/device/blob.hpp
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod/caffe-merge
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
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
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
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
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
<<<<<<< HEAD
<<<<<<< HEAD
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> origin/BVLC/parallel
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
>>>>>>> device-abstraction
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
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
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
      if (legacy_version) {
        loss += std::max(margin - dist_sq_.cpu_data()[i], Dtype(0.0));
      } else {
        Dtype dist = std::max(margin - sqrt(dist_sq_.cpu_data()[i]),
                              Dtype(0.0));
        loss += dist*dist;
      }
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
      loss += std::max(margin-dist_sq_.cpu_data()[i], Dtype(0.0));
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
>>>>>>> device-abstraction
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
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
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
=======
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
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
=======
>>>>>>> pod/device/blob.hpp
=======
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
=======
>>>>>>> pod/caffe-merge
=======
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
=======
>>>>>>> pod/device/blob.hpp
=======
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
=======
>>>>>>> pod/device/blob.hpp
=======
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
=======
>>>>>>> pod/device/blob.hpp
=======
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
=======
>>>>>>> pod/device/blob.hpp
__global__ void CLLForward(const int count, const int channels,
    const Dtype margin, const Dtype alpha,
>>>>>>> origin/BVLC/parallel
=======
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
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
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
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
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
>>>>>>> device-abstraction
    const Dtype* y, const Dtype* diff, const Dtype* dist_sq,
    Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    if (static_cast<int>(y[n])) {  // similar pairs
      bottom_diff[i] = alpha * diff[i];
    } else {  // dissimilar pairs
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
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
      Dtype mdist(0.0);
      Dtype beta(0.0);
      if (legacy_version) {
        mdist = (margin - dist_sq[n]);
        beta = -alpha;
      } else {
        Dtype dist = sqrt(dist_sq[n]);
        mdist = (margin - dist);
        beta = -alpha * mdist / (dist + Dtype(1e-4)) * diff[i];
      }
      if (mdist > 0.0) {
        bottom_diff[i] = beta;
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
      if ((margin-dist_sq[n]) > 0.0) {
        bottom_diff[i] = -alpha * diff[i];
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
>>>>>>> device-abstraction
      } else {
        bottom_diff[i] = 0;
      }
    }
  }
}

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int count = bottom[0]->count();
      const int channels = bottom[0]->channels();
      Dtype margin = this->layer_param_.contrastive_loss_param().margin();
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
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
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
<<<<<<< HEAD
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod/device/blob.hpp
=======
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod/caffe-merge
=======
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> origin/BVLC/parallel
=======
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
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
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> origin/BVLC/parallel
=======
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> origin/BVLC/parallel
=======
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
=======
>>>>>>> origin/BVLC/parallel
=======
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
>>>>>>> device-abstraction
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[0]->num());
      // NOLINT_NEXT_LINE(whitespace/operators)
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
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
=======
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
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
=======
>>>>>>> pod/device/blob.hpp
=======
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
=======
>>>>>>> pod/caffe-merge
=======
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
=======
>>>>>>> pod/device/blob.hpp
=======
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
=======
>>>>>>> pod/device/blob.hpp
=======
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
=======
>>>>>>> pod/device/blob.hpp
=======
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
=======
>>>>>>> pod/device/blob.hpp
      CLLForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, alpha,
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
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
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
>>>>>>> device-abstraction
          bottom[2]->gpu_data(),  // pair similarity 0 or 1
          diff_.gpu_data(),  // the cached eltwise difference between a and b
          dist_sq_.gpu_data(),  // the cached square distance between a and b
          bottom[i]->mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ContrastiveLossLayer);

}  // namespace caffe
