#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
__global__ void PReLUForward(const int n, int channels, int hw,
  const Dtype* in, Dtype* out, const Dtype* slope_data) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / hw) % channels;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}
// CUDA kernele for forward w/ channel shared
template <typename Dtype>
__global__ void PReLUChannelSharedForward(const int n, const Dtype* in,
  Dtype* out, const Dtype *slope_data) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[0];
  }
}

template <typename Dtype>
void PReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int hw = bottom[0]->height() * bottom[0]->width();
  const int channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->gpu_data();
  if (channel_shared_) {
    // Channel shared variant
    // NOLINT_NEXT_LINE(whitespace/operators)
    PReLUChannelSharedForward<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, slope_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    PReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, hw, bottom_data, top_data, slope_data);
    CUDA_POST_KERNEL_CHECK;
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void PReLUBackward(const int n, int channels, int hw,
  const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
  const Dtype* slope_data) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / hw) % channels;
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * slope_data[c]);
  }
}
// CUDA kernel for bottom backward w/ channel shared
template <typename Dtype>
__global__ void PReLUChannelSharedBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, const Dtype *slope_data) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * slope_data[0]);
  }
}
// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void PReLUParamBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * in_data[index] * (in_data[index] <= 0);
  }
}

template <typename Dtype>
void PReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int hw = bottom[0]->height() * bottom[0]->width();
  const int channels = bottom[0]->channels();
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* slope_data = this->blobs_[0]->gpu_data();
    if (channel_shared_) {
      // Channel shared variant
      // NOLINT_NEXT_LINE(whitespace/operators)
      PReLUChannelSharedBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, slope_data);
      CUDA_POST_KERNEL_CHECK;
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      PReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, hw, top_diff, bottom_data, bottom_diff, slope_data);
      CUDA_POST_KERNEL_CHECK;
    }
  }
  // Propagte to param
  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_gpu_diff();
    // slope_diff is set as 0, then accumulated over batches
    caffe_gpu_set<Dtype>(this->blobs_[0]->count(), Dtype(0), slope_diff);
    int chw = channels * hw;
    // compute element-wise diff
    Dtype dsum = 0.;
    for (int n = 0; n < bottom[0]->num(); ++n) {
      Dtype* temp_buff = multiplier_.mutable_gpu_diff();
      // NOLINT_NEXT_LINE(whitespace/operators)
      PReLUParamBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        chw, top_diff + top[0]->offset(n),
        bottom_data + bottom[0]->offset(n), multiplier_.mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
      if (channel_shared_) {
        // Channel shared variant
        Dtype d;
        caffe_gpu_dot<Dtype>(channels * hw, multiplier_.gpu_diff(),
          multiplier_.gpu_data(), &d);
        dsum += d;
      } else {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, hw, 1.,
          multiplier_.gpu_diff(), multiplier_.gpu_data(), 1.,
          slope_diff);
      }
    }
    if (channel_shared_) {
      caffe_gpu_set(this->blobs_[0]->count(), Dtype(dsum), slope_diff);
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(PReLULayer);


}  // namespace caffe
