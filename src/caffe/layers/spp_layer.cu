#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define DEBUG (debug && index >= 0 && index < 50)
#define debug 0
#define TOP 1
#define BOTTOM 1

namespace caffe {

template <typename Dtype>
__global__ void SPPForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int kernel_depth, const int output_size,
    Dtype* top_data, int* mask, Dtype* top_mask) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    int non_channel_index = index % output_size;
    int j = 1;
    // Shift holds the total of kernels at step i.
    int shift = 0;
    int next_shift = 1;
    while (non_channel_index >= next_shift) {
      shift = next_shift;
      ++j;
      next_shift += j * j;
    }
    int shifted_index = non_channel_index - shift;
    int num_pools = j;
    int pw = shifted_index % num_pools;
    int ph = (shifted_index / num_pools) % num_pools;
    int c = (index / output_size) % channels;
    int n = index / output_size / channels;
    if (DEBUG) {
      printf("Forward "
          "Index: %d\t"
          "Shifted Index: %d\t"
          "Num Pools: %d\t"
          "PW: %d\t"
          "PH: %d\t"
          "c: %d\t"
          "n: %d\t"
          "kernel_d: %d\t"
          "OS: %d\n", index, shifted_index, num_pools, pw, ph, c, n,
          kernel_depth, output_size);
    }
    int kernel_h = (height + num_pools - 1) / num_pools;
    int kernel_w = (width + num_pools - 1) / num_pools;
    int hstart = ph * kernel_h;
    int wstart = pw * kernel_w;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* bottom_data_shifted = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_data_shifted[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data_shifted[maxidx];
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
__global__ void SPPForwardWindowed(const int nthreads, const Dtype* bottom_data,
    const Dtype* bottom_window_data, const int num, const int channels,
    const int height, const int width, const int kernel_depth,
    const int output_size, const int window_count, const int image_w,
    const int image_h, Dtype* top_data, int* mask, Dtype* top_mask) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    int non_channel_index = index % output_size;
    int j = 1;
    // Shift holds the total number of pools at step j.
    int shift = 0;
    int next_shift = 1;
    while (non_channel_index >= next_shift) {
      shift = next_shift;
      ++j;
      next_shift += j * j;
    }
    int shifted_index = non_channel_index - shift;
    int num_pools = j;
    int pw = shifted_index % num_pools;
    int ph = shifted_index / num_pools;

    int win = (index / output_size) % window_count;
    int c = (index / output_size / window_count) % channels;
    int n = index / output_size / window_count / channels;

    // 4 = number of coordinates per row.
    const Dtype* bottom_window_data_shifted = bottom_window_data + (n * window_count + win)* 4;
    float window_x = bottom_window_data_shifted[0] * width / image_w;
    float window_y = bottom_window_data_shifted[1] * height / image_h;
    float window_w = max((bottom_window_data_shifted[2] * width / image_w), 1.0f);
    float window_h = max((bottom_window_data_shifted[3] * height / image_h), 1.0f);
    if (DEBUG && TOP) {
      printf("Window info: "
          "Index: %d\t"
          "win: %d\t"
          "wx: %f\t"
          "wy: %f\t"
          "ww: %f\t"
          "wh: %f\t"
          "width: %d\t"
          "height: %d\t"
          "window_count: %d\n",
          index,
          win,
          bottom_window_data[win*4],
          bottom_window_data[win*4+1],
          bottom_window_data[win*4+2],
          bottom_window_data[win*4+3],
          width,
          height,
          window_count);
    }


    // Using fractional heights to better represent smaller sections instead of
    // defaulting to repeating the end pixels over and over.
    float kernel_h = window_h / num_pools;
    float kernel_w = window_w / num_pools;
    float hstart_float = ph * kernel_h + window_y;
    float wstart_float = pw * kernel_w + window_x;
    // Starting point is floor of possible window.
    int hstart = (int)hstart_float;
    int wstart = (int)wstart_float;
    // Ending point is ceiling of possible window bounded by
    // dimensions of image and dimensions of patch.
    int hend = min(min((int)ceil(hstart_float + kernel_h), height), (int)ceil(window_y + window_h));
    int wend = min(min((int)ceil(wstart_float + kernel_w), width), (int)ceil(window_x + window_w));
    // Make sure over at least 1x1 patch for sure. Could have happened if hit
    // image size constraint I think.
    if (hstart == hend) {
      if (hend < height) {
        ++hend;
      } else {
        --hstart;
      }
    }
    if (wstart == wend) {
      if (wend < width) {
        ++wend;
      } else {
        --wstart;
      }
    }
    /*
    int mhend = MIN(f_hend, height);
    int mwend = MIN(f_wend, width);
    int hend = MIN(mhend, (int)(window_y + window_h));
    int wend = MIN(mwend, (int)(window_x + window_w));
    */
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;

    const Dtype* bottom_data_shifted = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_data_shifted[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data_shifted[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (DEBUG && TOP) {
      printf("Forward "
          "Index: %d\t"
          "Shifted Index: %d\t"
          "Num Pools: %d\t"
          "PW: %d\t"
          "PH: %d\t"
          "win: %d\t"
          "wstart: %d\t"
          "wend:  %d\t"
          "hstart: %d\t"
          "hend:  %d\t"
          "c: %d\t"
          "n: %d\t"
          "maxidx: %d\t"
          "maxval: %f"
          "\n", index, shifted_index, num_pools, pw, ph,
          win,
          wstart, wend,
          hstart, hend, c, n, maxidx, maxval);
    }
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
  if(debug) {
    printf("Forward_gpu\n");
  }
  if (bottom.size() > 1) {
    // Windowed case
    const Dtype* bottom_window_data = bottom[1]->gpu_data();
    SPPForwardWindowed<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom_window_data, bottom[0]->num(), channels_,
            height_, width_, kernel_depth_, output_size_, bottom[1]->height(),
            image_w_, image_h_, top_data, mask, top_mask);

  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    SPPForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, kernel_depth_, output_size_, top_data, mask, top_mask);
  }

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
    // Iterate through all positions in top array that could have resulted in
    // this index being the max.
    if (mask) {
      int shift = 0;
      for (int num_pools = 1; num_pools < kernel_depth; ++num_pools) {
        int pw = w / num_pools;
        int ph = h / num_pools;
        if (DEBUG) {
          printf("Mask Backward "
              "Index: %d\t"
              "w: %d\t"
              "h: %d\t"
              "num_pools : %d\t"
              "pw: %d\t"
              "ph: %d\t"
              "c: %d\t"
              "n: %d\t\n", index, w, h, num_pools, pw, ph,
              c, n);
        }
        if (mask[offset + shift + ph * num_pools + pw] == h * width + w) {
          gradient += top_diff[offset + shift + ph * num_pools + pw];
        }
        shift += num_pools * num_pools;
      }
    } else {
      int shift = 0;
      for (int num_pools = 1; num_pools < kernel_depth; ++num_pools) {
        int pw = w / num_pools;
        int ph = h / num_pools;
        if (DEBUG) {
            printf("No Mask Backward "
              "Index: %d\t"
              "w: %d\t"
              "h: %d\t"
              "num_pools : %d\t"
              "pw: %d\t"
              "ph: %d\t"
              "c: %d\t"
              "n: %d\t\n", index, w, h, num_pools, pw, ph,
              c, n);
        }
        if (top_mask[offset + shift + ph * num_pools + pw] == h * width + w) {
          gradient += top_diff[offset + shift + ph * num_pools + pw];
        }
        shift += num_pools * num_pools;
      }
    }
    bottom_diff[index] = gradient;
  }
}


// Float version should be the only one ever called.
__global__ void SPPBackwardWindowed(const int nthreads, const float* top_diff,
    const int* mask, const float* top_mask, const int num, const int channels,
    const int height, const int width, const int kernel_depth,
    const int output_size, const int window_count, float* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = (index / output_size / window_count) % channels;
    int n = index / output_size / window_count / channels;
    float* bottom_diff_shifted = bottom_diff + (n * channels + c) * height * width;
    if (mask) {
      atomicAdd(&bottom_diff_shifted[mask[index]], top_diff[index]);
    } else {
      atomicAdd(&bottom_diff_shifted[(int)top_mask[index]], top_diff[index]);
    }
    /*
    // Index in the bottom array.
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    Dtype gradient = 0;
    int offset = (n * channels + c) * output_size * window_count;
    // Iterate through all positions in top array that could have resulted in
    // this index being the max.
    if (mask) {
      for (int win = 0; win < window_count; ++win) {
        // Because of crazy indexing in windows, easier to check every window.
        for (int pool = 0; pool < output_size; ++pool) {
          if (mask[offset + win * output_size + pool] == h * width + w) {
            gradient += top_diff[offset + win * output_size + pool];
            if (DEBUG && BOTTOM) {
              printf("Mask Backward "
                "Index: %d\t"
                "w: %d\t"
                "h: %d\t"
                "c: %d\t"
                "n: %d\t"
                "pool: %d\t"
                "win: %d\t"
                "idx: %d\t"
                "gradient: %f\t"
                "top_diff[i]: %f\t"
                "mask[i]: %d\t"
                "\n", index, w, h, c, n, pool, win, win * output_size + pool, gradient, top_diff[win * output_size + pool], mask[offset + win * output_size + pool]);
            }
          }
        }
      }
    } else {
      for (int win = 0; win < window_count; ++win) {
        // Because of crazy indexing in windows, easier to check every window.
        for (int pool = 0; pool < output_size; ++pool) {
          if (top_mask[offset + win * output_size + pool] == h * width + w) {
            gradient += top_diff[offset + win * output_size + pool];
            if (DEBUG && BOTTOM) {
              printf("Mask Backward "
                "Index: %d\t"
                "w: %d\t"
                "h: %d\t"
                "c: %d\t"
                "n: %d\t"
                "i: %d\t"
                "win: %d\t"
                "idx: %d\t"
                "gradient: %f\t"
                "top_diff[i]: %f\t"
                "mask[i]: %d\t"
                "\n", index, w, h, c, n, pool, win, win * output_size + pool, gradient, top_diff[offset + win * output_size + pool], top_mask[offset + win * output_size + pool]);
            }
          }
        }
      }
    }
    bottom_diff[index] = gradient;
    if (DEBUG) {
      printf("Gradient: %f\n", bottom_diff[index]);
    }
  */
  }
}


__global__ void SPPBackwardWindowed(const int nthreads, const double* top_diff,
    const int* mask, const double* top_mask, const int num, const int channels,
    const int height, const int width, const int kernel_depth,
    const int output_size, const int window_count, double* bottom_diff) {
  printf("No double type implemented. You done fucked up.");
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
  const int top_count = top[0]->count();
  caffe_gpu_set(count, Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  if (use_top_mask) {
    top_mask = top[1]->gpu_data();
  } else {
    mask = max_idx_.gpu_data();
  }
  if (bottom.size() > 1) {
    const int window_count = bottom[1]->height();
    SPPBackwardWindowed<<<CAFFE_GET_BLOCKS(top_count),
        CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask, top_mask, top[0]->num(), channels_,
            height_, width_, kernel_depth_, output_size_, window_count,
            bottom_diff);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    SPPBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, kernel_depth_, output_size_, bottom_diff);
  }

  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SPPLayer);

}  // namespace caffe
