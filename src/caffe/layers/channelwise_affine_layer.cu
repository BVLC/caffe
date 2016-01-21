#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/channelwise_affine_layer.hpp"

namespace caffe {

// CUDA kernel for forward
template <typename Dtype>
__global__ void ChannelwiseAffineForward(const int n, const int channels,
    const int dim, const Dtype* in, Dtype* out, const Dtype* slope_data,
    const Dtype* bias_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] * slope_data[c] + bias_data[c];
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void ChannelwiseAffineBackward(const int n,
    const int channels, const int dim, const Dtype* in_diff,
    Dtype* out_diff, const Dtype* slope_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = slope_data[c] * in_diff[index];
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void ChannelwiseAffineParamSlopeBackward(const int n,
    const int rows, const int rowPitch, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
      out_diff[index] = in_diff[index] * in_data[index];
      for ( int k = 1; k < rows; k++ ) {
          out_diff[index] += in_diff[index + k*rowPitch]
          * in_data[index + k*rowPitch];
      }
  }
}

template <typename Dtype>
void ChannelwiseAffineLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->gpu_data();
  const Dtype* bias_data = this->blobs_[1]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  ChannelwiseAffineForward<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data,
      slope_data, bias_data, div_factor);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void ChannelwiseAffineLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int num = bottom[0]->shape(0);
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->shape(1);
  int height  = 1;
  int width = 1;
  if (bottom[0]->num_axes() > 2) {
    height = bottom[0]->shape(2);
    width = bottom[0]->shape(3);
  }

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }
  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      caffe_gpu_set(this->blobs_[1]->count(), Dtype(0.0), bias_diff);
    // Gradient with respect to bias
      for (int n = 0; n < num; ++n) {
          caffe_gpu_gemv<Dtype>(
            CblasNoTrans, channels, height * width, (Dtype)1.,
            top_diff + top[0]->offset(n), bias_multiplier_.gpu_data(),
            (Dtype)1., bias_diff);
      }
  }
  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_gpu_diff();
    int cdim = channels * dim;
      // compute element-wise diff
      // NOLINT_NEXT_LINE(whitespace/operators)
      ChannelwiseAffineParamSlopeBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
          CAFFE_CUDA_NUM_THREADS>>>(
          cdim, num, top[0]->offset(1), top_diff ,
          bottom_data,
          backward_buff_.mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
      if (channel_shared_) {
        Dtype d = 0;
        caffe_gpu_dot<Dtype>(cdim, backward_buff_.gpu_diff(),
            multiplier_.gpu_data(), &d);
        caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(d), slope_diff);
      } else {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, Dtype(1.),
             backward_buff_.gpu_diff(), multiplier_.gpu_data(), Dtype(1.),
            slope_diff);
      }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* slope_data = this->blobs_[0]->gpu_data();
    int div_factor = channel_shared_ ? channels : 1;
    // NOLINT_NEXT_LINE(whitespace/operators)
    ChannelwiseAffineBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, top_diff, bottom_diff, slope_data, div_factor);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ChannelwiseAffineLayer);

}  // namespace caffe
