/*
 * Axpy Layer
 *
 * Created on: May 1, 2017
 * Author: hujie
 */

#include "caffe/layers/axpy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void AxpyForward(const int count, const int spatial_dim, 
    const Dtype* scale_data, const Dtype* x_data, const Dtype* y_data,
    Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, count) {
    out_data[index] = scale_data[index / spatial_dim] * x_data[index]
        + y_data[index];
  }
}

template <typename Dtype>
void AxpyLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* scale_data = bottom[0]->gpu_data();
  const Dtype* x_data = bottom[1]->gpu_data();
  const Dtype* y_data = bottom[2]->gpu_data();
  Dtype* out_data = top[0]->mutable_gpu_data();
  const int count = bottom[1]->count();
  AxpyForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom[1]->count(2), scale_data, x_data, y_data, out_data);  
}

template <typename Dtype>
__global__ void AxpyBackwardScale(const int outer_num, const int spatial_dim, 
    const Dtype* x_data, const Dtype* top_diff, Dtype* scale_diff) {
  __shared__ Dtype buffer[CAFFE_CUDA_NUM_THREADS];
  unsigned int tid = threadIdx.x;
  buffer[tid] = 0;
  __syncthreads();

  for (int j = tid; j < spatial_dim; j += blockDim.x) {
    int offset = blockIdx.x * spatial_dim + j;
    buffer[tid] += top_diff[offset] * x_data[offset];
  }
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) {
      buffer[threadIdx.x] += buffer[threadIdx.x + i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    scale_diff[blockIdx.x] = buffer[0];
  }
}

template <typename Dtype>
__global__ void AxpyBackwardX(const int count, const int spatial_dim, 
    const Dtype* scale_data, const Dtype* top_diff, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = scale_data[index / spatial_dim] * top_diff[index];
  }
}

template <typename Dtype>
void AxpyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->gpu_diff();
  if (propagate_down[0]) {
    int outer_num = bottom[1]->count(0, 2);
    AxpyBackwardScale<Dtype><<<outer_num, CAFFE_CUDA_NUM_THREADS>>>(
        outer_num, bottom[1]->count(2),
        bottom[1]->gpu_data(), top_diff,
        bottom[0]->mutable_gpu_diff()); 
  }
  if (propagate_down[1]) {
    AxpyBackwardX<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top[0]->count(2), 
        bottom[0]->gpu_data(), top_diff, 
        bottom[1]->mutable_gpu_diff());
  }
  if (propagate_down[2]) {
    caffe_copy(count, top_diff, bottom[2]->mutable_gpu_diff());
  }
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(AxpyLayer);

}  // namespace caffe
