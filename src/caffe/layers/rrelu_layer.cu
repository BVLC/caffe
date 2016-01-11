#include <algorithm>
#include <vector>

#include "caffe/layers/rrelu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RReLUForward(const int n, const Dtype* in, Dtype* out,
    const Dtype *negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * Dtype(1.0) / negative_slope[index];
  }
}
template <typename Dtype>
__global__ void RReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

template <typename Dtype>
void RReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope_lower = this->layer_param_.rrelu_param().negative_slope_lower();
  Dtype negative_slope_upper = this->layer_param_.rrelu_param().negative_slope_upper();
  if (this->phase_ == TRAIN) {
    Dtype* mask =
        static_cast<Dtype*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_uniform(count, negative_slope_lower, negative_slope_upper,mask);  
  // NOLINT_NEXT_LINE(whitespace/operators)
  RReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, mask);
  CUDA_POST_KERNEL_CHECK;
  }
  else
  {
  Dtype negative_slope = Dtype(2)/(negative_slope_lower+negative_slope_upper);
  // NOLINT_NEXT_LINE(whitespace/operators)
  RReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_data, negative_slope);
  CUDA_POST_KERNEL_CHECK;
 }
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void RReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, const Dtype *negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * Dtype(1.0) / negative_slope[index]);
  }
}

template <typename Dtype>
__global__ void RReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
  }
}

template <typename Dtype>
void RReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope_lower = this->layer_param_.rrelu_param().negative_slope_lower();
    Dtype negative_slope_upper = this->layer_param_.rrelu_param().negative_slope_upper();
  if (this->phase_ == TRAIN) {
    Dtype* mask =
        static_cast<Dtype*>(rand_vec_.mutable_gpu_data());
	    // NOLINT_NEXT_LINE(whitespace/operators)
    RReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, mask);
    CUDA_POST_KERNEL_CHECK;
}
else
{
    Dtype negative_slope = Dtype(2)/(negative_slope_lower+negative_slope_upper);
    // NOLINT_NEXT_LINE(whitespace/operators)
    RReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, negative_slope);
    CUDA_POST_KERNEL_CHECK;
  }
  
}
}


INSTANTIATE_LAYER_GPU_FUNCS(RReLULayer);


}  // namespace caffe
