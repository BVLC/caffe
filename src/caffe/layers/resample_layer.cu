#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ResampleForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int num_output, Dtype* top_data, int* mask,
    Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int output_position = index % num_output;
    int c = (index / num_output ) % channels;
    int n = index / num_output / channels;
    int bottom_data_shift = (n * channels + c) * height * width;

    int sampled_index = output_position * width * height / num_output;
    top_data[index] = bottom_data[bottom_data_shift + sampled_index];
    if (mask) {
      mask[index] = sampled_index;
    } else {
      top_mask[index] = sampled_index;
    }
  }
}


template <typename Dtype>
__global__ void ResampleForwardWindowed(const int nthreads, const Dtype* bottom_data,
    const Dtype* bottom_window_data, const int num, const int channels,
    const int height, const int width, const int num_output,
    const int window_count, const int image_w, const int image_h,
    Dtype* top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int output_position = index % num_output;
    int win = (index / num_output) % window_count;
    int c = (index / num_output / window_count) % channels;
    int n = index / num_output / window_count / channels;
    int bottom_data_shift = (n * channels + c) * height * width;

    // 4 = number of coordinates per row.
    int wstart = bottom_window_data[(n + win) * 4] * width / image_w;
    int hstart = bottom_window_data[(n + win) * 4 + 1] * height / image_h;
    int window_w = max((int)ceil(bottom_window_data[(n + win) * 4 + 2] * width * 1.0 / image_w), 1);
    int window_h = max((int)ceil(bottom_window_data[(n + win) * 4 + 3] * height * 1.0 / image_h), 1);
    int wend = min(wstart + window_w, width);
    int hend = min(hstart + window_h, height);

    int patch_w = wend - wstart;
    int patch_h = hend - hstart;

    int new_index = output_position * patch_w * patch_h / num_output;
    int w_shift = new_index % patch_w;
    int h_shift = new_index / patch_w;

    int sampled_index = (hstart + h_shift) * width + wstart + w_shift;
    top_data[index] = bottom_data[bottom_data_shift + sampled_index];
    if (mask) {
      mask[index] = sampled_index;
    } else {
      top_mask[index] = sampled_index;
    }
  }
}


template <typename Dtype>
void ResampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
    mask = index_mask_.mutable_gpu_data();
  }
  if (bottom.size() > 1) {
    // Windowed case
    const Dtype* bottom_window_data = bottom[1]->gpu_data();
    ResampleForwardWindowed<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom_window_data, bottom[0]->num(), channels_,
            height_, width_, num_output_, bottom[1]->height(), image_w_,
            image_h_, top_data, mask, top_mask);

  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    ResampleForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, num_output_, top_data, mask, top_mask);
  }

  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void ResampleBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const Dtype* top_mask, const int num, const int channels,
    const int height, const int width, const int num_output, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  }
}


template <typename Dtype>
__global__ void ResampleBackwardWindowed(const int nthreads, const Dtype* top_diff,
    const int* mask, const Dtype* top_mask, const int num, const int channels,
    const int height, const int width, const int num_output,
    const int window_count, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // Index in the bottom array.
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    Dtype gradient = 0;
    int offset = (n * channels + c) * num_output * window_count;
    // Iterate through all positions in top array that could have resulted in
    // this index being the max.
    if (mask) {
      for (int win = 0; win < window_count; ++win) {
        // Because of crazy indexing in windows, easier to check every window.
        for (int pool = 0; pool < num_output; ++pool) {
          if (mask[offset + win * num_output + pool] == h * width + w) {
            gradient += top_diff[offset + win * num_output + pool];
          }
        }
      }
    } else {
      for (int win = 0; win < window_count; ++win) {
        // Because of crazy indexing in windows, easier to check every window.
        for (int pool = 0; pool < num_output; ++pool) {
          if (top_mask[offset + win * num_output + pool] == h * width + w) {
            gradient += top_diff[offset + win * num_output + pool];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void ResampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  if (use_top_mask) {
    top_mask = top[1]->gpu_data();
  } else {
    mask = index_mask_.gpu_data();
  }
  if (bottom.size() > 1) {
    const int window_count = bottom[1]->height();
    ResampleBackwardWindowed<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask, top_mask, top[0]->num(), channels_,
            height_, width_, num_output_, window_count, bottom_diff);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    ResampleBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask, top_mask, top[0]->num(), channels_,
            height_, width_, num_output_, bottom_diff);
  }

  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ResampleLayer);

}  // namespace caffe
