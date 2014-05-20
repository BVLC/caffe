// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

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
Dtype DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (Caffe::phase() == Caffe::TRAIN) {
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_->mutable_gpu_data());
    caffe_gpu_rng_uniform(count, mask);
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, mask, uint_thres_, scale_, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_gpu_copy(count, bottom_data, top_data);
  }
  return Dtype(0);
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
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  CHECK(Caffe::phase() == Caffe::TRAIN);
  if (propagate_down) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const unsigned int* mask =
        static_cast<const unsigned int*>(rand_vec_->gpu_data());
    const int count = (*bottom)[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, uint_thres_, scale_, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_CLASS(DropoutLayer);


}  // namespace caffe
