#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReshapeForward(
    const int nthreads, const Dtype* bottom_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    top_data[index] = bottom_data[index];
  }
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int bottom_count = bottom[0]->count();

  ReshapeForward<Dtype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
      bottom_count, bottom_data, top_data);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ReshapeBackward(const int nthreads, const Dtype* top_diff,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    bottom_diff[index] = top_diff[index];
  }
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  ReshapeBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, bottom_diff);
}
INSTANTIATE_LAYER_GPU_FUNCS(ReshapeLayer);

}  // namespace caffe
