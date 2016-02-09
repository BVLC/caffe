#include <algorithm>
#include <vector>

#include "caffe/layers/elu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ELUForward(const int n, const Dtype* in, Dtype* out,
    Dtype alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] :
        alpha * (exp(in[index]) - 1);
  }
}

template <typename Dtype>
void ELULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype alpha = this->layer_param_.elu_param().alpha();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ELUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, alpha);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ELUBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, const Dtype* in_data,
    Dtype* out_diff, Dtype alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_data[index] > 0 ? in_diff[index] :
        in_diff[index] * (out_data[index] + alpha);
  }
}

template <typename Dtype>
void ELULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype alpha = this->layer_param_.elu_param().alpha();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ELUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, bottom_data, bottom_diff, alpha);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ELULayer);


}  // namespace caffe
