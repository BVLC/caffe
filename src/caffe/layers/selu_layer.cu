#include <algorithm>
#include <vector>

#include "caffe/layers/selu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SeLUForward(const int n, const Dtype* in, Dtype* out,
    const Dtype alpha, const Dtype lambda) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? lambda*in[index] : lambda*alpha*(Dtype(exp(in[index]))-Dtype(1.));
  }
}

template <typename Dtype>
void SeLuLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  SeLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, alpha, lambda);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void SeLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, const Dtype alpha, const Dtype lambda) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_data[index] > 0 ? lambda*in_diff[index] : 
                        lambda*alpha*in_diff[index]*Dtype(exp(in_data[index]));
  }
}

template <typename Dtype>
void SeLuLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SeLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, alpha, lambda);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SeLuLayer);


}  // namespace caffe
