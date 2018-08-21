#include <vector>

#include "caffe/layers/clip_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

__global__ void ClipForward(const int n, const float* in, float* out,
    float p_min, float p_max) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = fmaxf(p_min, fminf(in[index], p_max));
  }
}

__global__ void ClipForward(const int n, const double* in, double* out,
    double p_min, double p_max) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = fmax(p_min, fmin(in[index], p_max));
  }
}

template <typename Dtype>
void ClipLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype p_min = this->layer_param_.clip_param().min();
  Dtype p_max = this->layer_param_.clip_param().max();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ClipForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, p_min, p_max);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ClipBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype p_min, Dtype p_max) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * (
            in_data[index] >= p_min && in_data[index] <= p_max);
  }
}

template <typename Dtype>
void ClipLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype p_min = this->layer_param_.clip_param().min();
    Dtype p_max = this->layer_param_.clip_param().max();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ClipBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, p_min, p_max);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ClipLayer);


}  // namespace caffe
