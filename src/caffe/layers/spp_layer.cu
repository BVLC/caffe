#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define debug 0

namespace caffe {

template <typename Dtype>
__global__ void SPPForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int kernel_depth, const int output_size,
    Dtype* top_data, int* mask, Dtype* top_mask) {

  // Shift holds the total of kernels at step i.
  CUDA_KERNEL_LOOP(index, nthreads) {
    int non_channel_index = index % output_size;
    int j = 0;
    int shift = 0;
    int next_shift = 1;
    while (non_channel_index >= next_shift) {
      shift = next_shift;
      ++j;
      next_shift += (1 << j) * (1 << j);
    }
    int shifted_index = non_channel_index - shift;
    int num_pools = 1 << j;
    int pw = shifted_index % num_pools;
    int ph = (shifted_index / num_pools) % num_pools;
    int c = (index / output_size) % channels;
    int n = index / output_size / channels;
    if (debug) {
      printf("Forwrard "
          "Index: %d\t"
          "Shifted Index: %d\t"
          "Num Pools: %d\t"
          "PW: %d\t"
          "PH: %d\t"
          "c: %d\t"
          "n: %d\t"
          "kernel_d: %d\t"
          "OS: %d\n", index, shifted_index, num_pools, pw, ph, c, n, kernel_depth, output_size);
    }
    int kernel_h = (height + num_pools - 1) / num_pools;
    int kernel_w = (width + num_pools - 1) / num_pools;
    int hstart = ph * kernel_h;
    int wstart = pw * kernel_w;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
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
void SPPLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  if (use_top_mask) {
    top_mask = top[1]->mutable_gpu_data();
  } else {
    mask = max_idx_.mutable_gpu_data();
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  SPPForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), channels_,
      height_, width_, kernel_depth_, output_size_, top_data, mask, top_mask);

  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void SPPBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const Dtype* top_mask, const int num, const int channels,
    const int height, const int width, const int kernel_depth,
    const int output_size, Dtype* bottom_diff) {
  // Shift holds the total of kernels at step i.
  CUDA_KERNEL_LOOP(index, nthreads) {
    // Index in the bottom array.
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    Dtype gradient = 0;
    int offset = (n * channels + c) * output_size;
    top_diff += offset;
    // Iterate through all positions in top array that could have resulted in this
    // index being the max.
    if (mask) {
      mask += offset;
      int shift = 0;
      for (int i = 0; i < kernel_depth; ++i) {
        int num_pools = 1 << i;
        int kernel_h = (height + num_pools - 1) / num_pools;
        int kernel_w = (width + num_pools - 1) / num_pools;
        int pw = w / num_pools;
        int ph = h / num_pools;
        if (debug) {
            printf("Mask Backward "
              "Index: %d\t"
              "w: %d\t"
              "h: %d\t"
              "num_pools : %d\t"
              "kernel_h: %d\t"
              "kernel_w: %d\t"
              "pw: %d\t"
              "ph: %d\t"
              "c: %d\t"
              "n: %d\t\n", index, w, h, num_pools, kernel_h, kernel_w, pw, ph, c, n);
        }
        if (mask[shift + ph * num_pools + pw] == h * width + w) {
          gradient += top_diff[ph * num_pools + pw];
        }
        shift += num_pools * num_pools;
      }
    } else {
      top_mask += offset;
      int shift = 0;
      for (int i = 0; i < kernel_depth; ++i) {
        int num_pools = 1 << i;
        int kernel_h = (height + num_pools - 1) / num_pools;
        int kernel_w = (width + num_pools - 1) / num_pools;
        int pw = w / num_pools;
        int ph = h / num_pools;
        if (debug) {
            printf("No Mask Backward "
              "Index: %d\t"
              "w: %d\t"
              "h: %d\t"
              "num_pools : %d\t"
              "kernel_h: %d\t"
              "kernel_w: %d\t"
              "pw: %d\t"
              "ph: %d\t"
              "c: %d\t"
              "n: %d\t\n", index, w, h, num_pools, kernel_h, kernel_w, pw, ph, c, n);
        }
        if (top_mask[shift + ph * num_pools + pw] == h * width + w) {
          gradient += top_diff[ph * num_pools + pw];
        }
        shift += num_pools * num_pools;
      }
    }
  }

}

template <typename Dtype>
void SPPLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
  if (use_top_mask) {
    top_mask = top[1]->gpu_data();
  } else {
    mask = max_idx_.gpu_data();
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  SPPBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, mask, top_mask, top[0]->num(), channels_,
      height_, width_, kernel_depth_, output_size_, bottom_diff);

  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SPPLayer);

}  // namespace caffe
