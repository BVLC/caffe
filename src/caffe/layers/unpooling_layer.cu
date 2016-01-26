#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/unpooling_layer.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void MaxUnpoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype* mask, const Dtype* argmax_count,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_size, const int stride, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int pool_channel_offset = (n * channels + c) * pooled_height * pooled_width;
    int pool_index = pool_channel_offset + ph * pooled_width + pw;
    int channel_offset = (n * channels + c) * height * width;
    const int top_index = channel_offset + static_cast<int>(mask[pool_index]);
    if (argmax_count) {
      const Dtype unpooled_act =
          bottom_data[pool_index] / argmax_count[top_index];
      caffe_gpu_atomic_add(unpooled_act, top_data + top_index);
    } else {
      top_data[top_index] = bottom_data[pool_index];
    }
  }
}

template <typename Dtype>
__global__ void AveUnpoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype* pool_count, const int num, const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int kernel_size, const int stride, const int pad, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width + pad;
    int h = (index / width) % height + pad;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = (h < kernel_size) ? 0 : (h - kernel_size) / stride + 1;
    int phend = min(h / stride + 1, pooled_height);
    int pwstart = (w < kernel_size) ? 0 : (w - kernel_size) / stride + 1;
    int pwend = min(w / stride + 1, pooled_width);
    Dtype top_datum = 0;
    bottom_data += (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        top_datum += bottom_data[ph * pooled_width + pw];
      }
    }
    top_data[index] = top_datum / pool_count[h * width + w];
  }
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const bool overlapping = (stride_ < kernel_size_);
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  const Dtype* mask;
  const Dtype* pool_count;
  const Dtype* argmax_count = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    mask = bottom[1]->gpu_data();
    if (overlapping) {
      argmax_count = bottom[2]->gpu_data();
    }
    caffe_gpu_set(count, Dtype(0), top_data);
    count = bottom[0]->count();
    MaxUnpoolForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, mask, argmax_count, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_size_, stride_,
        top_data);
    break;
  case PoolingParameter_PoolMethod_AVE:
    pool_count = pool_count_.gpu_data();
    AveUnpoolForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, pool_count, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_size_, stride_,
        pad_, top_data);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxUnpoolBackward(const int nthreads,
    const Dtype* top_diff, const Dtype* argmax_count,
    const Dtype* mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_size, const int stride,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int top_offset = (n * channels + c) * height * width;
    int top_index = top_offset + static_cast<int>(mask[index]);
    int num_top_use = argmax_count ? argmax_count[top_index] : 1;
    assert(num_top_use != 0);
    bottom_diff[index] = top_diff[top_index] / num_top_use;
  }
}

template <typename Dtype>
__global__ void AveUnpoolBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* pool_count, const int num, const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const int top_offset = (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += top_diff[top_offset + h * width + w] /
                  pool_count[h * width + w];
      }
    }
    bottom_diff[index] = aveval;
  }
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const bool overlapping = (stride_ < kernel_size_);
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  const Dtype* mask;
  const Dtype* argmax_count = NULL;
  const Dtype* pool_count;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    mask = bottom[1]->gpu_data();
    if (overlapping) {
      argmax_count = bottom[2]->gpu_data();
    }
    MaxUnpoolBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, argmax_count, mask, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_size_, stride_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    pool_count = pool_count_.gpu_data();
    AveUnpoolBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, pool_count, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_size_,
        kernel_size_, stride_, stride_, pad_, pad_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(UnpoolingLayer);

}  // namespace caffe
