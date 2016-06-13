#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/equiv_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void EquivMaxPoolForward(const int nthreads,
    const Dtype* bottom_data, const int num, const int channels,
    const int height, const int width, const int equiv_pooled_height,
    const int equiv_pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % equiv_pooled_width;
    int ph = (index / equiv_pooled_width) % equiv_pooled_height;
    int c = (index / equiv_pooled_width / equiv_pooled_height) % channels;
    int n = index / equiv_pooled_width / equiv_pooled_height / channels;

    int hstart = ph - pad_h;
    int wstart = pw - pad_w;
    int hend = hstart + (stride_h * (kernel_h - 1) + 1);
    int wend = wstart + (stride_w * (kernel_w - 1) + 1);

    while (hstart < 0)
      hstart += stride_h;
    while (wstart < 0)
      wstart += stride_w;
    while (hend > height)
      hend -= stride_h;
    while (wend > width)
      wend -= stride_w;

    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; h += stride_h) {
      for (int w = wstart; w < wend; w += stride_w) {
        if (bottom_data[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}



template <typename Dtype>
void EquivPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.equiv_pooling_param().pool()) {
  case EquivPoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    EquivMaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data,
        bottom[0]->num(), channels_, height_, width_,
        equiv_pooled_height_, equiv_pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
        mask, top_mask);
    break;
  default:
    LOG(FATAL) << "Unknown equiv_pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void EquivMaxPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const Dtype* top_mask, const int num, const int channels,
    const int height, const int width, const int equiv_pooled_height,
    const int equiv_pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int phstart = (h + pad_h - (stride_h * (kernel_h - 1) + 1)) + 1;
    if (phstart < 0)
      phstart = 0;

    int phend = (h + pad_h) + 1;
    if (phend > equiv_pooled_height)
      phend = equiv_pooled_height;

    int pwstart = (w + pad_w - (stride_w * (kernel_w - 1) + 1)) + 1;
    if (pwstart < 0)
      pwstart = 0;

    int pwend = (w + pad_w) + 1;
    if (pwend > equiv_pooled_width)
      pwend = equiv_pooled_width;

    Dtype gradient = 0;
    int offset = (n * channels + c) * equiv_pooled_height * equiv_pooled_width;
    top_diff += offset;
    if (mask) {
      mask += offset;
      for (int ph = phstart; ph < phend; ph += 1) {
        for (int pw = pwstart; pw < pwend; pw += 1) {
          if (mask[ph * equiv_pooled_width + pw] == h * width + w) {
            gradient += top_diff[ph * equiv_pooled_width + pw];
          }
        }
      }
    } else {
      top_mask += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask[ph * equiv_pooled_width + pw] == h * width + w) {
            gradient += top_diff[ph * equiv_pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void EquivPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.equiv_pooling_param().pool()) {
  case EquivPoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    EquivMaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, mask,
        top_mask, top[0]->num(), channels_, height_, width_,
        equiv_pooled_height_, equiv_pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown equiv_pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(EquivPoolingLayer);

}  // namespace caffe
