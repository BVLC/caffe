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
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
#include "caffe/neuron_layers.hpp"
=======
#include "caffe/layers/dropout_layer.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/layers/dropout_layer.hpp"
>>>>>>> BVLC/master
#include "caffe/util/math_functions.hpp"
=======
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
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
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
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/vision_layers.hpp"
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
=======
>>>>>>> BVLC/device-abstraction
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> BVLC/device-abstraction
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
=======
=======
>>>>>>> pod/device/blob.hpp
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> device-abstraction
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
<<<<<<< HEAD
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
>>>>>>> pod/caffe-merge
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
>>>>>>> pod/device/blob.hpp
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

namespace caffe {

template <typename Dtype>
__global__ void DropoutForward(const int n, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    CURAND_CHECK(curandGenerate(Caffe::curand_generator(), mask, count));
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, mask, uint_thres_, scale_, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    GetDevice<Dtype>(Caffe::GPU)->copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void DropoutBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
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
    if (this->phase_ == TRAIN) {
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
    if (this->phase_ == TRAIN) {
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
    if (this->phase_ == TRAIN) {
=======
>>>>>>> pod-caffe-pod.hpp-merge
    if (Caffe::phase() == Caffe::TRAIN) {
>>>>>>> origin/BVLC/parallel
=======
    if (this->phase_ == TRAIN) {
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
=======
>>>>>>> pod/caffe-merge
=======
    if (this->phase_ == TRAIN) {
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
    if (this->phase_ == TRAIN) {
=======
    if (Caffe::phase() == Caffe::TRAIN) {
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
    if (this->phase_ == TRAIN) {
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
    if (this->phase_ == TRAIN) {
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
    if (this->phase_ == TRAIN) {
=======
    if (Caffe::phase() == Caffe::TRAIN) {
>>>>>>> origin/BVLC/parallel
=======
    if (this->phase_ == TRAIN) {
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
    if (this->phase_ == TRAIN) {
=======
    if (Caffe::phase() == Caffe::TRAIN) {
>>>>>>> origin/BVLC/parallel
=======
    if (this->phase_ == TRAIN) {
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
=======
    if (this->phase_ == TRAIN) {
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
      const unsigned int* mask =
          static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)
      DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, mask, uint_thres_, scale_, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    } else {
      GetDevice<Dtype>(Caffe::GPU)->copy(top[0]->count(), top_diff,
                                         bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DropoutLayer);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge

>>>>>>> pod-caffe-pod.hpp-merge

}  // namespace caffe
