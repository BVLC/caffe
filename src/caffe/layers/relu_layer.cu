// Copyright 2013 Yangqing Jia

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include <algorithm>

using std::max;

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = max(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
Dtype ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * (bottom_data[i] >= 0);
    }
  }
  return Dtype(0);
}

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    out[index] = max(in[index], Dtype(0.));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    out_diff[index] = in_diff[index] * (in_data[index] >= 0);
  }
}

template <typename Dtype>
Dtype ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count();
    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
  return Dtype(0);
}

INSTANTIATE_CLASS(ReLULayer);


}  // namespace caffe
