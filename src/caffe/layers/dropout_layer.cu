// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/vision_layers.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::SetUp(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.reset(new SyncedMemory(bottom[0]->count() * sizeof(int)));
  threshold_ = this->layer_param_.dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = (unsigned int)(UINT_MAX * threshold_);
};

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  int* mask = (int*)rand_vec_->mutable_cpu_data();
  const int count = bottom[0]->count();
  if (Caffe::phase() == Caffe::TRAIN) {
    // Create random numbers
    //viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, Caffe::vsl_stream(),
    //    count, mask, 1. - threshold_);
    caffe_vRngBernoulli<int>(count, mask, 1. - threshold_);

    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    memcpy(top_data, bottom_data, bottom[0]->count() * sizeof(Dtype));
  }
}

template <typename Dtype>
Dtype DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  CHECK(Caffe::phase() == Caffe::TRAIN);
  if (propagate_down) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int* mask = (int*)(rand_vec_->cpu_data());
    const int count = (*bottom)[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * mask[i] * scale_;
    }
  }
  return Dtype(0);
}

template <typename Dtype>
__global__ void DropoutForward(const int n, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (Caffe::phase() == Caffe::TRAIN) {
    CURAND_CHECK(curandGenerate(Caffe::curand_generator(),
        (unsigned int*)(rand_vec_->mutable_gpu_data()), count));
    // set thresholds
    DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, (unsigned int*)rand_vec_->gpu_data(), uint_thres_, scale_,
        top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    CUDA_CHECK(cudaMemcpy(top_data, bottom_data,
        count * sizeof(Dtype), cudaMemcpyDeviceToDevice));
  }
}

template <typename Dtype>
__global__ void DropoutBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}

template <typename Dtype>
Dtype DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  CHECK(Caffe::phase() == Caffe::TRAIN);
  if (propagate_down) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const unsigned int* mask = (unsigned int*)rand_vec_->gpu_data();
    const int count = (*bottom)[0]->count();
    DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, uint_thres_, scale_, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
  return Dtype(0);
}

INSTANTIATE_CLASS(DropoutLayer);


}  // namespace caffe
