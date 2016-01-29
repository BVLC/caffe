#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif  // USE_GREENTEA

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void MaxPoolForward(const int_tp nthreads,
                               const Dtype* const bottom_data, const int_tp num,
                               const int_tp channels, const int_tp height,
                               const int_tp width, const int_tp pooled_height,
                               const int_tp pooled_width, const int_tp kernel_h,
                               const int_tp kernel_w, const int_tp stride_h,
                               const int_tp stride_w, const int_tp pad_h,
                               const int_tp pad_w, Dtype* const top_data,
                               int_tp* mask,
                               Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int_tp pw = index % pooled_width;
    const int_tp ph = (index / pooled_width) % pooled_height;
    const int_tp c = (index / pooled_width / pooled_height) % channels;
    const int_tp n = index / pooled_width / pooled_height / channels;
    int_tp hstart = ph * stride_h - pad_h;
    int_tp wstart = pw * stride_w - pad_w;
    const int_tp hend = min((int_tpc) (hstart + kernel_h), (int_tpc) height);
    const int_tp wend = min((int_tpc) (wstart + kernel_w), (int_tpc) width);
    hstart = max((int_tpc) (hstart), (int_tpc) (0));
    wstart = max((int_tpc) (wstart), (int_tpc) (0));
    Dtype maxval = -FLT_MAX;
    int_tp maxidx = -1;
    const Dtype* const bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
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

template<typename Dtype>
__global__ void AvePoolForward(const int_tp nthreads,
                               const Dtype* const bottom_data, const int_tp num,
                               const int_tp channels, const int_tp height,
                               const int_tp width, const int_tp pooled_height,
                               const int_tp pooled_width, const int_tp kernel_h,
                               const int_tp kernel_w, const int_tp stride_h,
                               const int_tp stride_w, const int_tp pad_h,
                               const int_tp pad_w, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int_tp pw = index % pooled_width;
    const int_tp ph = (index / pooled_width) % pooled_height;
    const int_tp c = (index / pooled_width / pooled_height) % channels;
    const int_tp n = index / pooled_width / pooled_height / channels;
    int_tp hstart = ph * stride_h - pad_h;
    int_tp wstart = pw * stride_w - pad_w;
    int_tp hend = min((int_tpc) (hstart + kernel_h),
                      (int_tpc) (height + pad_h));
    int_tp wend = min((int_tpc) (wstart + kernel_w), (int_tpc) (width + pad_w));
    const int_tp pool_size = (hend - hstart) * (wend - wstart);
    hstart = max((int_tpc) (hstart), (int_tpc) (0));
    wstart = max((int_tpc) (wstart), (int_tpc) (0));
    hend = min((int_tpc) (hend), (int_tpc) (height));
    wend = min((int_tpc) (wend), (int_tpc) (width));
    Dtype aveval = 0;
    const Dtype* const bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

template<typename Dtype>
__global__ void StoPoolForwardTrain(const int_tp nthreads,
                                    const Dtype* const bottom_data,
                                    const int_tp num, const int_tp channels,
                                    const int_tp height, const int_tp width,
                                    const int_tp pooled_height,
                                    const int_tp pooled_width,
                                    const int_tp kernel_h,
                                    const int_tp kernel_w,
                                    const int_tp stride_h,
                                    const int_tp stride_w,
                                    Dtype* const rand_idx,
                                    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int_tp pw = index % pooled_width;
    const int_tp ph = (index / pooled_width) % pooled_height;
    const int_tp c = (index / pooled_width / pooled_height) % channels;
    const int_tp n = index / pooled_width / pooled_height / channels;
    const int_tp hstart = ph * stride_h;
    const int_tp hend = min((int_tpc) (hstart + kernel_h), (int_tpc) height);
    const int_tp wstart = pw * stride_w;
    const int_tp wend = min((int_tpc) (wstart + kernel_w), (int_tpc) width);
    Dtype cumsum = 0.;
    const Dtype* const bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    // First pass: get sum
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
      }
    }
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_slice[h * width + w];
          return;
        }
      }
    }
  }
}

template<typename Dtype>
__global__ void StoPoolForwardTest(const int_tp nthreads,
                                   const Dtype* const bottom_data,
                                   const int_tp num, const int_tp channels,
                                   const int_tp height, const int_tp width,
                                   const int_tp pooled_height,
                                   const int_tp pooled_width,
                                   const int_tp kernel_h, const int_tp kernel_w,
                                   const int_tp stride_h, const int_tp stride_w,
                                   Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int_tp pw = index % pooled_width;
    const int_tp ph = (index / pooled_width) % pooled_height;
    const int_tp c = (index / pooled_width / pooled_height) % channels;
    const int_tp n = index / pooled_width / pooled_height / channels;
    const int_tp hstart = ph * stride_h;
    const int_tp hend = min((int_tpc) (hstart + kernel_h), (int_tpc) height);
    const int_tp wstart = pw * stride_w;
    const int_tp wend = min((int_tpc) (wstart + kernel_w), (int_tpc) width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    const Dtype* const bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    // First pass: get sum
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}

template<typename Dtype>
__global__ void MaxPoolBackward(const int_tp nthreads,
                                const Dtype* const top_diff,
                                const int_tp* const mask,
                                const Dtype* const top_mask, const int_tp num,
                                const int_tp channels, const int_tp height,
                                const int_tp width, const int_tp pooled_height,
                                const int_tp pooled_width,
                                const int_tp kernel_h, const int_tp kernel_w,
                                const int_tp stride_h, const int_tp stride_w,
                                const int_tp pad_h, const int_tp pad_w,
                                Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp c = (index / width / height) % channels;
    const int_tp n = index / width / height / channels;
    const int_tp phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int_tp phend = min((int_tpc) ((h + pad_h) / stride_h + 1L),
                             (int_tpc) pooled_height);
    const int_tp pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int_tp pwend = min((int_tpc) ((w + pad_w) / stride_w + 1L),
                             (int_tpc) pooled_width);
    Dtype gradient = 0;
    const int_tp offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int_tp* const mask_slice = mask + offset;
      for (int_tp ph = phstart; ph < phend; ++ph) {
        for (int_tp pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      const Dtype* const top_mask_slice = top_mask + offset;
      for (int_tp ph = phstart; ph < phend; ++ph) {
        for (int_tp pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template<typename Dtype>
__global__ void AvePoolBackward(const int_tp nthreads,
                                const Dtype* const top_diff, const int_tp num,
                                const int_tp channels, const int_tp height,
                                const int_tp width, const int_tp pooled_height,
                                const int_tp pooled_width,
                                const int_tp kernel_h, const int_tp kernel_w,
                                const int_tp stride_h, const int_tp stride_w,
                                const int_tp pad_h, const int_tp pad_w,
                                Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int_tp w = index % width + pad_w;
    const int_tp h = (index / width) % height + pad_h;
    const int_tp c = (index / width / height) % channels;
    const int_tp n = index / width / height / channels;
    const int_tp phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int_tp phend = min((int_tpc) (h / stride_h + 1),
                             (int_tpc) (pooled_height));
    const int_tp pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int_tp pwend = min((int_tpc) (w / stride_w + 1),
                             (int_tpc) (pooled_width));
    Dtype gradient = 0;
    const Dtype* const top_diff_slice = top_diff
        + (n * channels + c) * pooled_height * pooled_width;
    for (int_tp ph = phstart; ph < phend; ++ph) {
      for (int_tp pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int_tp hstart = ph * stride_h - pad_h;
        int_tp wstart = pw * stride_w - pad_w;
        int_tp hend = min((int_tpc) (hstart + kernel_h),
                          (int_tpc) (height + pad_h));
        int_tp wend = min((int_tpc) (wstart + kernel_w),
                          (int_tpc) (width + pad_w));
        int_tp pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

template<typename Dtype>
__global__ void StoPoolBackward(const int_tp nthreads,
                                const Dtype* const rand_idx,
                                const Dtype* const top_diff, const int_tp num,
                                const int_tp channels, const int_tp height,
                                const int_tp width, const int_tp pooled_height,
                                const int_tp pooled_width,
                                const int_tp kernel_h, const int_tp kernel_w,
                                const int_tp stride_h, const int_tp stride_w,
                                Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp c = (index / width / height) % channels;
    const int_tp n = index / width / height / channels;
    const int_tp phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int_tp phend = min((int_tpc) (h / stride_h + 1),
                             (int_tpc) pooled_height);
    const int_tp pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int_tp pwend = min((int_tpc) (w / stride_w + 1),
                             (int_tpc) pooled_width);
    Dtype gradient = 0;
    const Dtype* const rand_idx_slice = rand_idx
        + (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff
        + (n * channels + c) * pooled_height * pooled_width;
    for (int_tp ph = phstart; ph < phend; ++ph) {
      for (int_tp pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff_slice[ph * pooled_width + pw]
            * (index
                == static_cast<int_tp>(rand_idx_slice[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}

template<typename Dtype>
__global__ void MaxPoolForward(const int_tp nthreads, const Dtype* bottom_data,
                               const int_tp num, const int_tp channels,
                               const int_tp height, const int_tp width,
                               const int_tp pooled_height,
                               const int_tp pooled_width, const int_tp kernel_h,
                               const int_tp kernel_w, const int_tp ext_kernel_h,
                               const int_tp ext_kernel_w, const int_tp stride_h,
                               const int_tp stride_w, const int_tp dilation_h,
                               const int_tp dilation_w, const int_tp pad_h,
                               const int_tp pad_w, Dtype* top_data,
                               int_tp* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int_tp pw = index % pooled_width;
    int_tp ph = (index / pooled_width) % pooled_height;
    int_tp c = (index / pooled_width / pooled_height) % channels;
    int_tp n = index / pooled_width / pooled_height / channels;
    int_tp hstart = ph * stride_h - pad_h;
    int_tp wstart = pw * stride_w - pad_w;
    int_tp hend = min((int_tpc) (hstart + ext_kernel_h), (int_tpc) height);
    int_tp wend = min((int_tpc) (wstart + ext_kernel_w), (int_tpc) width);
    hstart = max((int_tpc) hstart, (int_tpc) (0));
    wstart = max((int_tpc) wstart, (int_tpc) (0));
    Dtype maxval = -FLT_MAX;
    int_tp maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int_tp h = hstart; h < hend; h += dilation_h) {
      for (int_tp w = wstart; w < wend; w += dilation_w) {
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

template<typename Dtype>
__global__ void AvePoolForward(const int_tp nthreads, const Dtype* bottom_data,
                               const int_tp num, const int_tp channels,
                               const int_tp height, const int_tp width,
                               const int_tp pooled_height,
                               const int_tp pooled_width, const int_tp kernel_h,
                               const int_tp kernel_w, const int_tp ext_kernel_h,
                               const int_tp ext_kernel_w, const int_tp stride_h,
                               const int_tp stride_w, const int_tp dilation_h,
                               const int_tp dilation_w, const int_tp pad_h,
                               const int_tp pad_w, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int_tp pw = index % pooled_width;
    int_tp ph = (index / pooled_width) % pooled_height;
    int_tp c = (index / pooled_width / pooled_height) % channels;
    int_tp n = index / pooled_width / pooled_height / channels;
    int_tp hstart = ph * stride_h - pad_h;
    int_tp wstart = pw * stride_w - pad_w;
    int_tp hend = min((int_tpc) (hstart + ext_kernel_h),
                      (int_tpc) (height + pad_h));
    int_tp wend = min((int_tpc) (wstart + ext_kernel_w),
                      (int_tpc) (width + pad_w));
    hstart = max((int_tpc) hstart, (int_tpc) (0));
    wstart = max((int_tpc) wstart, (int_tpc) (0));
    hend = min((int_tpc) hend, (int_tpc) height);
    wend = min((int_tpc) wend, (int_tpc) width);
    Dtype aveval = 0;
    bottom_data += (n * channels + c) * height * width;
    int_tp pool_size = 0;
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
        aveval += bottom_data[h * width + w];
        ++pool_size;
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

template<typename Dtype>
__global__ void StoPoolForwardTrain(const int_tp nthreads,
                                    const Dtype* bottom_data, const int_tp num,
                                    const int_tp channels, const int_tp height,
                                    const int_tp width,
                                    const int_tp pooled_height,
                                    const int_tp pooled_width,
                                    const int_tp kernel_h,
                                    const int_tp kernel_w,
                                    const int_tp ext_kernel_h,
                                    const int_tp ext_kernel_w,
                                    const int_tp stride_h,
                                    const int_tp stride_w,
                                    const int_tp dilation_h,
                                    const int_tp dilation_w, Dtype* rand_idx,
                                    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int_tp pw = index % pooled_width;
    int_tp ph = (index / pooled_width) % pooled_height;
    int_tp c = (index / pooled_width / pooled_height) % channels;
    int_tp n = index / pooled_width / pooled_height / channels;
    int_tp hstart = ph * stride_h;
    int_tp hend = min((int_tpc) (hstart + ext_kernel_h), (int_tpc) height);
    int_tp wstart = pw * stride_w;
    int_tp wend = min((int_tpc) (wstart + ext_kernel_w), (int_tpc) width);
    Dtype cumsum = 0.;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int_tp h = hstart; h < hend; h += dilation_h) {
      for (int_tp w = wstart; w < wend; w += dilation_w) {
        cumsum += bottom_data[h * width + w];
      }
    }
    float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int_tp h = hstart; h < hend; h += dilation_h) {
      for (int_tp w = wstart; w < wend; w += dilation_w) {
        cumsum += bottom_data[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_data[h * width + w];
          return;
        }
      }
    }
  }
}

template<typename Dtype>
__global__ void StoPoolForwardTest(const int_tp nthreads,
                                   const Dtype* bottom_data, const int_tp num,
                                   const int_tp channels, const int_tp height,
                                   const int_tp width,
                                   const int_tp pooled_height,
                                   const int_tp pooled_width,
                                   const int_tp kernel_h, const int_tp kernel_w,
                                   const int_tp ext_kernel_h,
                                   const int_tp ext_kernel_w,
                                   const int_tp stride_h, const int_tp stride_w,
                                   const int_tp dilation_h,
                                   const int_tp dilation_w, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int_tp pw = index % pooled_width;
    int_tp ph = (index / pooled_width) % pooled_height;
    int_tp c = (index / pooled_width / pooled_height) % channels;
    int_tp n = index / pooled_width / pooled_height / channels;
    int_tp hstart = ph * stride_h;
    int_tp hend = min((int_tpc) (hstart + ext_kernel_h), (int_tpc) height);
    int_tp wstart = pw * stride_w;
    int_tp wend = min((int_tpc) (wstart + ext_kernel_w), (int_tpc) width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int_tp h = hstart; h < hend; h += dilation_h) {
      for (int_tp w = wstart; w < wend; w += dilation_w) {
        cumsum += bottom_data[h * width + w];
        cumvalues += bottom_data[h * width + w] * bottom_data[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}

template<typename Dtype>
__global__ void MaxPoolBackward(const int_tp nthreads, const Dtype* top_diff,
                                const int_tp* mask, const Dtype* top_mask,
                                const int_tp num, const int_tp channels,
                                const int_tp height, const int_tp width,
                                const int_tp pooled_height,
                                const int_tp pooled_width,
                                const int_tp kernel_h, const int_tp kernel_w,
                                const int_tp ext_kernel_h,
                                const int_tp ext_kernel_w,
                                const int_tp stride_h, const int_tp stride_w,
                                const int_tp dilation_h,
                                const int_tp dilation_w, const int_tp pad_h,
                                const int_tp pad_w,
                                Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int_tp w = index % width;
    int_tp h = (index / width) % height;
    int_tp c = (index / width / height) % channels;
    int_tp n = index / width / height / channels;

    int_tp pooled_height_1 = pooled_height - 1;
    int_tp pooled_width_1 = pooled_width - 1;
    int_tp phstart =
        (h < ext_kernel_h) ? h % dilation_h : (h - ext_kernel_h) + 1;
    int_tp phend =
        (h >= pooled_height) ?
            pooled_height_1 - (pooled_height_1 - phstart) % dilation_h : h;
    int_tp pwstart =
        (w < ext_kernel_w) ? w % dilation_w : (w - ext_kernel_w) + 1;
    int_tp pwend =
        (w >= pooled_width) ?
            pooled_width_1 - (pooled_width_1 - pwstart) % dilation_w : w;

    Dtype gradient = 0;
    int_tp offset = (n * channels + c) * pooled_height * pooled_width;
    top_diff += offset;
    if (mask) {
      mask += offset;
      for (int_tp ph = phstart; ph <= phend; ph += dilation_h) {
        for (int_tp pw = pwstart; pw <= pwend; pw += dilation_w) {
          if (mask[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff[ph * pooled_width + pw];
          }
        }
      }
    } else {
      top_mask += offset;
      for (int_tp ph = phstart; ph <= phend; ph += dilation_h) {
        for (int_tp pw = pwstart; pw <= pwend; pw += dilation_w) {
          if (top_mask[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template<typename Dtype>
__global__ void MaxPoolNDForward(const int_tp n, const int_tp num_axes,
                                 const Dtype* bottom_data,
                                 const int_tp channels, const int_tp* size,
                                 const int_tp* pooled_size,
                                 const int_tp* kernel_size,
                                 const int_tp* ext_kernel_size,
                                 const int_tp* stride, const int_tp* dilation,
                                 const int_tp* pad, Dtype* top_data,
                                 int_tp* mask, Dtype* top_mask) {
  int_tp d_idx[6];  // NOLINT(runtime/arrays)
  int_tp d_start[6];  // NOLINT(runtime/arrays)
  int_tp d_end[6];  // NOLINT(runtime/arrays)
  int_tp d_iter[6];  // NOLINT(runtime/arrays)
  int_tp i;

  CUDA_KERNEL_LOOP(index, n) {
    int_tp offset = 1;
    int_tp num = index;
    for (i = num_axes - 1; i >= 0; --i) {
      d_idx[i] = num % pooled_size[i];
      d_start[i] = d_idx[i] * stride[i] - pad[i];
      d_end[i] = min((int_tpc) (d_start[i] + ext_kernel_size[i]),
                     (int_tpc) (size[i]));
      d_start[i] = max((int_tpc) (d_start[i]), (int_tpc) (0));
      num /= pooled_size[i];
      offset *= size[i];
      d_iter[i] = d_start[i];

      if (d_start[i] >= d_end[i]) {
        top_data[index] = -FLT_MAX;
        if (mask) {
          mask[index] = -1;
        } else {
          top_mask[index] = -1;
        }
        return;
      }
    }
    int_tp chan = num % channels;
    num /= channels;
    offset *= (num * channels + chan);

    Dtype maxval = -FLT_MAX;
    int_tp maxidx = -1;
    int_tp final_offset = 0;

    bool incremented;
    do {
      final_offset = offset;
      int_tp size_prod = 1;
      for (i = num_axes - 1; i >= 0; --i) {
        final_offset += d_iter[i] * size_prod;
        size_prod *= size[i];
      }

      if (bottom_data[final_offset] > maxval) {
        maxidx = final_offset;
        maxval = bottom_data[maxidx];
      }

      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        if (d_iter[i] >= d_end[i] - dilation[i]) {
          d_iter[i] = d_start[i];
        } else {
          d_iter[i] += dilation[i];
          incremented = true;
          break;
        }
      }
    } while (incremented);

    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template<typename Dtype>
__global__ void MaxPoolNDBackward(const int_tp n, const int_tp num_axes,
                                  const Dtype* top_diff, const int_tp* mask,
                                  const Dtype* top_mask, const int_tp channels,
                                  const int_tp* size, const int_tp* pooled_size,
                                  const int_tp* kernel_size,
                                  const int_tp* ext_kernel_size,
                                  const int_tp* stride, const int_tp* dilation,
                                  const int_tp* pad, Dtype* bottom_diff) {
  int_tp d_idx[6];  // NOLINT(runtime/arrays)
  int_tp d_start[6];  // NOLINT(runtime/arrays)
  int_tp d_end[6];  // NOLINT(runtime/arrays)
  int_tp d_iter[6];  // NOLINT(runtime/arrays)
  int_tp i;

  CUDA_KERNEL_LOOP(index, n) {
    // find out the local index
    // find out the local offset
    int_tp offset = 1;
    int_tp num = index;
    for (i = num_axes - 1; i >= 0; --i) {
      d_idx[i] = num % size[i];
      if (dilation[i] > 1) {
        d_start[i] =
            (d_idx[i] < ext_kernel_size[i]) ?
                d_idx[i] % dilation[i] : (d_idx[i] - ext_kernel_size[i]) + 1;
        d_end[i] =
            (d_idx[i] >= pooled_size[i]) ?
                (pooled_size[i] - 1)
                    - (pooled_size[i] - 1 - d_start[i]) % dilation[i] :
                d_idx[i];
      } else {
        d_start[i] =
            (d_idx[i] + pad[i] < kernel_size[i]) ?
                0 : (d_idx[i] + pad[i] - kernel_size[i]) / stride[i] + 1;
        d_end[i] = min((int_tpc) ((d_idx[i] + pad[i]) / stride[i] + 1),
                       (int_tpc) (pooled_size[i]));
      }
      num /= size[i];
      offset *= pooled_size[i];
      d_iter[i] = d_start[i];

      if (d_start[i] > d_end[i]) {
        bottom_diff[index] = 0;
        return;
      }
    }

    int_tp chan = num % channels;
    num /= channels;
    offset *= (num * channels + chan);

    Dtype gradient = 0;
    int_tp final_offset = 0;
    int_tp im_offset = 0;

    bool incremented;
    do {
      final_offset = offset;
      im_offset = 0;
      int_tp size_prod = 1;
      int_tp pooled_size_prod = 1;
      for (i = num_axes - 1; i >= 0; --i) {
        final_offset += d_iter[i] * pooled_size_prod;
        im_offset += d_idx[i] * size_prod;
        size_prod *= size[i];
        pooled_size_prod *= pooled_size[i];
      }
      if (mask) {
        if (mask[final_offset] == im_offset) {
          gradient += top_diff[final_offset];
        }
      } else {
        if (top_mask[final_offset] == im_offset) {
          gradient += top_diff[final_offset];
        }
      }

      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        if (d_iter[i] > d_end[i] - dilation[i]) {
          d_iter[i] = d_start[i];
        } else {
          d_iter[i] += dilation[i];
          incremented = true;
          break;
        }
      }
    } while (incremented);
    bottom_diff[index] = gradient;
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int_tp count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int_tp* mask = NULL;
  Dtype* top_mask = NULL;

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA

    if (num_spatial_axes_ == 2) {
      int_tp kernel_h_ = kernel_shape_.cpu_data()[0];
      int_tp kernel_w_ = kernel_shape_.cpu_data()[1];
      int_tp stride_h_ = stride_.cpu_data()[0];
      int_tp stride_w_ = stride_.cpu_data()[1];
      int_tp pad_h_ = pad_.cpu_data()[0];
      int_tp pad_w_ = pad_.cpu_data()[1];
      int_tp dilation_h_ = dilation_.cpu_data()[0];
      int_tp dilation_w_ = dilation_.cpu_data()[1];
      int_tp height_ = size_.cpu_data()[0];
      int_tp width_ = size_.cpu_data()[1];
      int_tp pooled_height_ = pooled_size_.cpu_data()[0];
      int_tp pooled_width_ = pooled_size_.cpu_data()[1];
      int_tp ext_kernel_h = ext_kernel_shape_.cpu_data()[0];
      int_tp ext_kernel_w = ext_kernel_shape_.cpu_data()[0];

      // 2D case
      if (use_skernel_) {
        // 2D-SK case
        switch (this->layer_param_.pooling_param().pool()) {
          case PoolingParameter_PoolMethod_MAX:
            if (use_top_mask) {
              top_mask = top[1]->mutable_gpu_data();
            } else {
              mask = max_idx_.mutable_gpu_data();
            }
            // NOLINT_NEXT_LINE(whitespace/operators)
            MaxPoolForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                CAFFE_CUDA_NUM_THREADS)(
                count, bottom_data, bottom[0]->shape(0), channels_,
                height_, width_, pooled_height_, pooled_width_, kernel_h_,
                kernel_w_, ext_kernel_h, ext_kernel_w,
                stride_h_, stride_w_, dilation_h_, dilation_w_,
                pad_h_, pad_w_, top_data,
                mask, top_mask);
            break;
          case PoolingParameter_PoolMethod_AVE:
            // NOLINT_NEXT_LINE(whitespace/operators)
            AvePoolForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                CAFFE_CUDA_NUM_THREADS)(
                count, bottom_data, bottom[0]->shape(0), channels_,
                height_, width_, pooled_height_, pooled_width_, kernel_h_,
                kernel_w_, ext_kernel_h, ext_kernel_w,
                stride_h_, stride_w_, dilation_h_, dilation_w_,
                pad_h_, pad_w_, top_data);
            break;
          case PoolingParameter_PoolMethod_STOCHASTIC:
            if (this->phase_ == caffe::TRAIN) {
              // We need to create the random index as well.
              caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                                    rand_idx_.mutable_gpu_data());
              // NOLINT_NEXT_LINE(whitespace/operators)
              StoPoolForwardTrain<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                  CAFFE_CUDA_NUM_THREADS)(
                  count, bottom_data, bottom[0]->shape(0), channels_,
                  height_, width_, pooled_height_, pooled_width_, kernel_h_,
                  kernel_w_, ext_kernel_h, ext_kernel_w,
                  stride_h_, stride_w_, dilation_h_, dilation_w_,
                  rand_idx_.mutable_gpu_data(), top_data);
            } else {
              // NOLINT_NEXT_LINE(whitespace/operators)
              StoPoolForwardTest<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                  CAFFE_CUDA_NUM_THREADS)(
                  count, bottom_data, bottom[0]->shape(0), channels_,
                  height_, width_, pooled_height_, pooled_width_, kernel_h_,
                  kernel_w_, ext_kernel_h, ext_kernel_w,
                  stride_h_, stride_w_, dilation_h_, dilation_w_, top_data);
            }
            break;
          default: {
            LOG(FATAL)<< "Unknown pooling method.";
          }
        }
        CUDA_POST_KERNEL_CHECK;
      } else {
        // 2D case
        switch (this->layer_param_.pooling_param().pool()) {
          case PoolingParameter_PoolMethod_MAX:
          if (use_top_mask) {
            top_mask = top[1]->mutable_gpu_data();
          } else {
            mask = max_idx_.mutable_gpu_data();
          }
          // NOLINT_NEXT_LINE(whitespace/operators)
          MaxPoolForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
              CAFFE_CUDA_NUM_THREADS)(
              count, bottom_data, bottom[0]->shape(0), channels_,
              height_, width_, pooled_height_, pooled_width_, kernel_h_,
              kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
              mask, top_mask);
          break;
          case PoolingParameter_PoolMethod_AVE:
          // NOLINT_NEXT_LINE(whitespace/operators)
          AvePoolForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
              CAFFE_CUDA_NUM_THREADS)(
              count, bottom_data, bottom[0]->shape(0), channels_,
              height_, width_, pooled_height_, pooled_width_, kernel_h_,
              kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
          break;
          case PoolingParameter_PoolMethod_STOCHASTIC:
          if (this->phase_ == TRAIN) {
            // We need to create the random index as well.
            caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                rand_idx_.mutable_gpu_data());
            // NOLINT_NEXT_LINE(whitespace/operators)
            StoPoolForwardTrain<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                CAFFE_CUDA_NUM_THREADS)(
                count, bottom_data, bottom[0]->shape(0), channels_,
                height_, width_, pooled_height_, pooled_width_, kernel_h_,
                kernel_w_, stride_h_, stride_w_,
                rand_idx_.mutable_gpu_data(), top_data);
          } else {
            // NOLINT_NEXT_LINE(whitespace/operators)
            StoPoolForwardTest<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                CAFFE_CUDA_NUM_THREADS)(
                count, bottom_data, bottom[0]->shape(0), channels_,
                height_, width_, pooled_height_, pooled_width_, kernel_h_,
                kernel_w_, stride_h_, stride_w_, top_data);
          }
          break;
          default: {
            LOG(FATAL)<< "Unknown pooling method.";
          }
        }
        CUDA_POST_KERNEL_CHECK;
      }
    } else {
      switch (this->layer_param_.pooling_param().pool()) {
        case PoolingParameter_PoolMethod_MAX:
        if (use_top_mask) {
          top_mask = top[1]->mutable_gpu_data();
        } else {
          mask = max_idx_.mutable_gpu_data();
        }
        // NOLINT_NEXT_LINE(whitespace/operators)
        MaxPoolNDForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS)(
            count, num_spatial_axes_, bottom_data,
            channels_, size_.gpu_data(), pooled_size_.gpu_data(),
            kernel_shape_.gpu_data(), ext_kernel_shape_.gpu_data(),
            stride_.gpu_data(), dilation_.gpu_data(), pad_.gpu_data(),
            top_data, mask, top_mask);
        break;
        default: {
          LOG(FATAL)<< "Unknown pooling method.";
        }
      }
    }
    CUDA_POST_KERNEL_CHECK;

#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    if (num_spatial_axes_ == 2) {
      int_tp kernel_h_ = kernel_shape_.cpu_data()[0];
      int_tp kernel_w_ = kernel_shape_.cpu_data()[1];
      int_tp stride_h_ = stride_.cpu_data()[0];
      int_tp stride_w_ = stride_.cpu_data()[1];
      int_tp pad_h_ = pad_.cpu_data()[0];
      int_tp pad_w_ = pad_.cpu_data()[1];
      int_tp dilation_h_ = dilation_.cpu_data()[0];
      int_tp dilation_w_ = dilation_.cpu_data()[1];
      int_tp height_ = size_.cpu_data()[0];
      int_tp width_ = size_.cpu_data()[1];
      int_tp pooled_height_ = pooled_size_.cpu_data()[0];
      int_tp pooled_width_ = pooled_size_.cpu_data()[1];
      int_tp ext_kernel_h = ext_kernel_shape_.cpu_data()[0];
      int_tp ext_kernel_w = ext_kernel_shape_.cpu_data()[0];

      // 2D case
      if (use_skernel_) {
        // 2D-SK case
        switch (this->layer_param_.pooling_param().pool()) {
          case PoolingParameter_PoolMethod_MAX: {
            if (use_top_mask) {
              top_mask = top[1]->mutable_gpu_data();
            } else {
              mask = max_idx_.mutable_gpu_data();
            }
            viennacl::ocl::kernel &oclk_max_pool_forward = program.get_kernel(
                CL_KERNEL_SELECT("max_pool_forward_sk"));
            viennacl::ocl::enqueue(
                oclk_max_pool_forward(count,
                    WrapHandle((cl_mem) bottom_data, &ctx),
                    bottom[0]->shape(0), channels_, height_, width_,
                    pooled_height_, pooled_width_, kernel_h_,
                    kernel_w_, ext_kernel_h, ext_kernel_w,
                    stride_h_, stride_w_, dilation_h_, dilation_w_,
                    pad_h_, pad_w_,
                    WrapHandle((cl_mem) top_data, &ctx),
                    mask == NULL ? 0 : 1,
                    WrapHandle((cl_mem) mask, &ctx),
                    WrapHandle((cl_mem) top_mask, &ctx)),
                ctx.get_queue());
          }
          break;
          case PoolingParameter_PoolMethod_AVE: {
            viennacl::ocl::kernel &oclk_ave_pool_forward = program.get_kernel(
                CL_KERNEL_SELECT("ave_pool_forward_sk"));
            viennacl::ocl::enqueue(
                oclk_ave_pool_forward(count,
                    WrapHandle((cl_mem) bottom_data, &ctx),
                    bottom[0]->shape(0), channels_,
                    height_, width_, pooled_height_, pooled_width_, kernel_h_,
                    kernel_w_, ext_kernel_h, ext_kernel_w,
                    stride_h_, stride_w_, dilation_h_, dilation_w_,
                    pad_h_, pad_w_, WrapHandle((cl_mem)top_data, &ctx)),
                ctx.get_queue());
          }
          break;
          case PoolingParameter_PoolMethod_STOCHASTIC: {
            if (this->phase_ == caffe::TRAIN) {
              // We need to create the random index as well.
              greentea_gpu_rng_uniform(this->device_->id(), count,
                  Dtype(0), Dtype(1),
                  (cl_mem)(rand_idx_.mutable_gpu_data()), 0);

              viennacl::ocl::kernel &oclk_sto_pool_forward = program.get_kernel(
                  CL_KERNEL_SELECT("sto_pool_forward_train_sk"));
              viennacl::ocl::enqueue(
                  oclk_sto_pool_forward(count,
                      WrapHandle((cl_mem)bottom_data, &ctx),
                      bottom[0]->shape(0), channels_,
                      height_, width_, pooled_height_, pooled_width_, kernel_h_,
                      kernel_w_, ext_kernel_h, ext_kernel_w,
                      stride_h_, stride_w_, dilation_h_, dilation_w_,
                      WrapHandle((cl_mem)(rand_idx_.mutable_gpu_data()), &ctx),
                      WrapHandle((cl_mem)(top_data), &ctx)),
                  ctx.get_queue());
            } else {
              viennacl::ocl::kernel &oclk_sto_pool_forward = program.get_kernel(
                  CL_KERNEL_SELECT("sto_pool_forward_test_sk"));
              viennacl::ocl::enqueue(
                  oclk_sto_pool_forward(count,
                      WrapHandle((cl_mem)bottom_data, &ctx),
                      bottom[0]->shape(0), channels_,
                      height_, width_, pooled_height_, pooled_width_, kernel_h_,
                      kernel_w_, ext_kernel_h, ext_kernel_w,
                      stride_h_, stride_w_, dilation_h_, dilation_w_,
                      WrapHandle((cl_mem)top_data, &ctx)),
                  ctx.get_queue());
            }
          }
          break;
          default: {
            LOG(FATAL)<< "Unknown pooling method.";
          }
        }
      } else {
        // 2D case
        switch (this->layer_param_.pooling_param().pool()) {
          case PoolingParameter_PoolMethod_MAX: {
            if (use_top_mask) {
              top_mask = top[1]->mutable_gpu_data();
            } else {
              mask = max_idx_.mutable_gpu_data();
            }
            viennacl::ocl::kernel &oclk_max_pool_forward = program.get_kernel(
                CL_KERNEL_SELECT("max_pool_forward"));
            viennacl::ocl::enqueue(
                oclk_max_pool_forward(count,
                    WrapHandle((cl_mem) bottom_data, &ctx),
                    bottom[0]->shape(0), channels_, height_, width_,
                    pooled_height_, pooled_width_, kernel_h_,
                    kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
                    WrapHandle((cl_mem) top_data, &ctx),
                    mask == NULL ? 0 : 1,
                    WrapHandle((cl_mem) mask, &ctx),
                    WrapHandle((cl_mem) top_mask, &ctx)),
                ctx.get_queue());
          }
          break;
          case PoolingParameter_PoolMethod_AVE: {
            viennacl::ocl::kernel &oclk_ave_pool_forward = program.get_kernel(
                CL_KERNEL_SELECT("ave_pool_forward"));
            viennacl::ocl::enqueue(
                oclk_ave_pool_forward(count,
                    WrapHandle((cl_mem) bottom_data, &ctx),
                    bottom[0]->shape(0), channels_,
                    height_, width_, pooled_height_, pooled_width_, kernel_h_,
                    kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
                    WrapHandle((cl_mem)top_data, &ctx)),
                ctx.get_queue());
          }
          break;
          case PoolingParameter_PoolMethod_STOCHASTIC: {
            if (this->phase_ == caffe::TRAIN) {
              // We need to create the random index as well.
              greentea_gpu_rng_uniform(this->device_->id(), count,
                  Dtype(0), Dtype(1),
                  (cl_mem)(rand_idx_.mutable_gpu_data()), 0);

              viennacl::ocl::kernel &oclk_sto_pool_forward = program.get_kernel(
                  CL_KERNEL_SELECT("sto_pool_forward_train"));
              viennacl::ocl::enqueue(
                  oclk_sto_pool_forward(count,
                      WrapHandle((cl_mem)bottom_data, &ctx),
                      bottom[0]->shape(0), channels_,
                      height_, width_, pooled_height_, pooled_width_,
                      kernel_h_, kernel_w_,
                      stride_h_, stride_w_,
                      WrapHandle((cl_mem)(rand_idx_.mutable_gpu_data()), &ctx),
                      WrapHandle((cl_mem)top_data, &ctx)),
                  ctx.get_queue());
            } else {
              viennacl::ocl::kernel &oclk_sto_pool_forward = program.get_kernel(
                  CL_KERNEL_SELECT("sto_pool_forward_test"));
              viennacl::ocl::enqueue(
                  oclk_sto_pool_forward(count,
                      WrapHandle((cl_mem)bottom_data, &ctx),
                      bottom[0]->shape(0), channels_,
                      height_, width_, pooled_height_,
                      pooled_width_, kernel_h_, kernel_w_,
                      stride_h_, stride_w_, WrapHandle((cl_mem)top_data, &ctx)),
                  ctx.get_queue());
            }
          }
          break;
          default: {
            LOG(FATAL)<< "Unknown pooling method.";
          }
        }
      }
    } else {
      switch (this->layer_param_.pooling_param().pool()) {
        case PoolingParameter_PoolMethod_MAX: {
          if (use_top_mask) {
            top_mask = top[1]->mutable_gpu_data();
          } else {
            mask = max_idx_.mutable_gpu_data();
          }
          viennacl::ocl::kernel &oclk_max_pool_forward = program.get_kernel(
              CL_KERNEL_SELECT("max_pool_forward_nd"));
          viennacl::ocl::enqueue(
              oclk_max_pool_forward(count, num_spatial_axes_,
                  WrapHandle((cl_mem)bottom_data, &ctx),
                  channels_,
                  WrapHandle((cl_mem)(size_.gpu_data()), &ctx),
                  WrapHandle((cl_mem)(pooled_size_.gpu_data()), &ctx),
                  WrapHandle((cl_mem)(kernel_shape_.gpu_data()), &ctx),
                  WrapHandle((cl_mem)(ext_kernel_shape_.gpu_data()), &ctx),
                  WrapHandle((cl_mem)(stride_.gpu_data()), &ctx),
                  WrapHandle((cl_mem)(dilation_.gpu_data()), &ctx),
                  WrapHandle((cl_mem)(pad_.gpu_data()), &ctx),
                  WrapHandle((cl_mem)top_data, &ctx),
                  mask == NULL ? 0 : 1,
                  WrapHandle((cl_mem)mask, &ctx),
                  WrapHandle((cl_mem)top_mask, &ctx)),
              ctx.get_queue());
        }
        break;
        default: {
          LOG(FATAL)<< "Unknown pooling method.";
        }
      }
    }

#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int_tp count = bottom[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int_tp* mask = NULL;
  const Dtype* top_mask = NULL;

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    caffe_gpu_set(count, Dtype(0.), bottom_diff);

    if (num_spatial_axes_ == 2) {
      int_tp kernel_h_ = kernel_shape_.cpu_data()[0];
      int_tp kernel_w_ = kernel_shape_.cpu_data()[1];
      int_tp stride_h_ = stride_.cpu_data()[0];
      int_tp stride_w_ = stride_.cpu_data()[1];
      int_tp pad_h_ = pad_.cpu_data()[0];
      int_tp pad_w_ = pad_.cpu_data()[1];
      int_tp dilation_h_ = dilation_.cpu_data()[0];
      int_tp dilation_w_ = dilation_.cpu_data()[1];
      int_tp height_ = size_.cpu_data()[0];
      int_tp width_ = size_.cpu_data()[1];
      int_tp pooled_height_ = pooled_size_.cpu_data()[0];
      int_tp pooled_width_ = pooled_size_.cpu_data()[1];
      int_tp ext_kernel_h = ext_kernel_shape_.cpu_data()[0];
      int_tp ext_kernel_w = ext_kernel_shape_.cpu_data()[0];

      if (use_skernel_) {
        switch (this->layer_param_.pooling_param().pool()) {
          case PoolingParameter_PoolMethod_MAX:
            if (use_top_mask) {
              top_mask = top[1]->gpu_data();
            } else {
              mask = max_idx_.gpu_data();
            }
            // NOLINT_NEXT_LINE(whitespace/operators)
            MaxPoolBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                CAFFE_CUDA_NUM_THREADS)(
                count, top_diff, mask, top_mask, top[0]->shape(0), channels_,
                height_, width_, pooled_height_, pooled_width_,
                kernel_h_, kernel_w_, ext_kernel_h, ext_kernel_w,
                stride_h_, stride_w_, dilation_h_, dilation_w_,
                pad_h_, pad_w_,
                bottom_diff);
            break;
          default:
            LOG(FATAL)<<
            "Unknown or unsupported pooling method in Backward_gpu().";
          }
          CUDA_POST_KERNEL_CHECK;
        } else {
          switch (this->layer_param_.pooling_param().pool()) {
            case PoolingParameter_PoolMethod_MAX:
            if (use_top_mask) {
              top_mask = top[1]->gpu_data();
            } else {
              mask = max_idx_.gpu_data();
            }
            // NOLINT_NEXT_LINE(whitespace/operators)
            MaxPoolBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                CAFFE_CUDA_NUM_THREADS)(
                count, top_diff, mask, top_mask, top[0]->shape(0), channels_,
                height_, width_, pooled_height_, pooled_width_,
                kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
                bottom_diff);
            break;
            case PoolingParameter_PoolMethod_AVE:
            // NOLINT_NEXT_LINE(whitespace/operators)
            AvePoolBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                CAFFE_CUDA_NUM_THREADS)(
                count, top_diff, top[0]->shape(0), channels_,
                height_, width_, pooled_height_, pooled_width_, kernel_h_,
                kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
            break;
            case PoolingParameter_PoolMethod_STOCHASTIC:
            // NOLINT_NEXT_LINE(whitespace/operators)
            StoPoolBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                CAFFE_CUDA_NUM_THREADS)(
                count, rand_idx_.gpu_data(), top_diff,
                top[0]->shape(0), channels_, height_, width_, pooled_height_,
                pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
                bottom_diff);
            break;
            default: {
              LOG(FATAL)<< "Unknown pooling method.";
            }
          }
          CUDA_POST_KERNEL_CHECK;
        }
      } else {
        switch (this->layer_param_.pooling_param().pool()) {
          case PoolingParameter_PoolMethod_MAX:
          if (use_top_mask) {
            top_mask = top[1]->gpu_data();
          } else {
            mask = max_idx_.gpu_data();
          }
          // NOLINT_NEXT_LINE(whitespace/operators)
          MaxPoolNDBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
              CAFFE_CUDA_NUM_THREADS)(
              count, num_spatial_axes_, top_diff, mask, top_mask,
              channels_, size_.gpu_data(), pooled_size_.gpu_data(),
              kernel_shape_.gpu_data(), ext_kernel_shape_.gpu_data(),
              stride_.gpu_data(), dilation_.gpu_data(), pad_.gpu_data(),
              bottom_diff);
          break;
          default:
          LOG(FATAL)<<
          "Unknown or unsupported pooling method in Backward_gpu().";
        }
        CUDA_POST_KERNEL_CHECK;
      }
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(
          this->device_->id());
      viennacl::ocl::program &program = this->device_->program();

      greentea_gpu_set(this->device_->id(), count, Dtype(0.),
          (cl_mem) bottom_diff, 0);

      if (num_spatial_axes_ == 2) {
        int_tp kernel_h_ = kernel_shape_.cpu_data()[0];
        int_tp kernel_w_ = kernel_shape_.cpu_data()[1];
        int_tp stride_h_ = stride_.cpu_data()[0];
        int_tp stride_w_ = stride_.cpu_data()[1];
        int_tp pad_h_ = pad_.cpu_data()[0];
        int_tp pad_w_ = pad_.cpu_data()[1];
        int_tp dilation_h_ = dilation_.cpu_data()[0];
        int_tp dilation_w_ = dilation_.cpu_data()[1];
        int_tp height_ = size_.cpu_data()[0];
        int_tp width_ = size_.cpu_data()[1];
        int_tp pooled_height_ = pooled_size_.cpu_data()[0];
        int_tp pooled_width_ = pooled_size_.cpu_data()[1];
        int_tp ext_kernel_h = ext_kernel_shape_.cpu_data()[0];
        int_tp ext_kernel_w = ext_kernel_shape_.cpu_data()[0];

        if (use_skernel_) {
          switch (this->layer_param_.pooling_param().pool()) {
            case PoolingParameter_PoolMethod_MAX: {
              if (use_top_mask) {
                top_mask = top[1]->gpu_data();
              } else {
                mask = max_idx_.gpu_data();
              }
              viennacl::ocl::kernel &oclk_max_pool_backward =
              program.get_kernel(
                  CL_KERNEL_SELECT("max_pool_backward_sk"));
              viennacl::ocl::enqueue(
                  oclk_max_pool_backward(count,
                      WrapHandle((cl_mem) top_diff, &ctx),
                      mask == NULL ? 0 : 1,
                      WrapHandle((cl_mem) mask, &ctx),
                      WrapHandle((cl_mem) top_mask, &ctx),
                      top[0]->shape(0), channels_, height_, width_,
                      pooled_height_, pooled_width_, kernel_h_,
                      kernel_w_, ext_kernel_h, ext_kernel_w,
                      stride_h_, stride_w_, dilation_h_, dilation_w_,
                      pad_h_, pad_w_,
                      WrapHandle((cl_mem) bottom_diff, &ctx)),
                  ctx.get_queue());
            }
            break;
            default:
            LOG(FATAL)<<
            "Unknown or unsupported pooling method in Backward_gpu().";
          }
        } else {
          switch (this->layer_param_.pooling_param().pool()) {
            case PoolingParameter_PoolMethod_MAX: {
              if (use_top_mask) {
                top_mask = top[1]->gpu_data();
              } else {
                mask = max_idx_.gpu_data();
              }
              viennacl::ocl::kernel &oclk_max_pool_backward =
              program.get_kernel(
                  CL_KERNEL_SELECT("max_pool_backward"));
              viennacl::ocl::enqueue(
                  oclk_max_pool_backward(count,
                      WrapHandle((cl_mem) top_diff, &ctx),
                      mask == NULL ? 0 : 1,
                      WrapHandle((cl_mem) mask, &ctx),
                      WrapHandle((cl_mem) top_mask, &ctx),
                      top[0]->shape(0), channels_, height_, width_,
                      pooled_height_, pooled_width_, kernel_h_,
                      kernel_w_, stride_h_, stride_w_, pad_h_,
                      pad_w_,
                      WrapHandle((cl_mem) bottom_diff, &ctx)),
                  ctx.get_queue());
            }
            break;
            case PoolingParameter_PoolMethod_AVE: {
              viennacl::ocl::kernel &oclk_ave_pool_backward =
              program.get_kernel(
                  CL_KERNEL_SELECT("ave_pool_backward"));
              viennacl::ocl::enqueue(
                  oclk_ave_pool_backward(count,
                      WrapHandle((cl_mem) top_diff, &ctx),
                      top[0]->shape(0), channels_, height_, width_,
                      pooled_height_, pooled_width_, kernel_h_,
                      kernel_w_, stride_h_, stride_w_, pad_h_,
                      pad_w_,
                      WrapHandle((cl_mem) bottom_diff, &ctx)),
                  ctx.get_queue());
            }
            break;
            case PoolingParameter_PoolMethod_STOCHASTIC: {
              viennacl::ocl::kernel &oclk_sto_pool_backward =
              program.get_kernel(
                  CL_KERNEL_SELECT("sto_pool_backward"));
              viennacl::ocl::enqueue(
                  oclk_sto_pool_backward(
                      count, WrapHandle((cl_mem) (rand_idx_.gpu_data()), &ctx),
                      WrapHandle((cl_mem) top_diff, &ctx), top[0]->shape(0),
                      channels_,
                      height_, width_, pooled_height_, pooled_width_, kernel_h_,
                      kernel_w_, stride_h_, stride_w_,
                      WrapHandle((cl_mem) bottom_diff, &ctx)),
                  ctx.get_queue());
            }
            break;
            default: {
              LOG(FATAL)<< "Unknown pooling method.";
            }
          }
        }
      } else {
        switch (this->layer_param_.pooling_param().pool()) {
          case PoolingParameter_PoolMethod_MAX: {
            if (use_top_mask) {
              top_mask = top[1]->gpu_data();
            } else {
              mask = max_idx_.gpu_data();
            }
            viennacl::ocl::kernel &oclk_max_pool_backward = program.get_kernel(
                CL_KERNEL_SELECT("max_pool_backward_nd"));
            viennacl::ocl::enqueue(
                oclk_max_pool_backward(
                    count, num_spatial_axes_,
                    WrapHandle((cl_mem) top_diff, &ctx),
                    mask == NULL ? 0 : 1, WrapHandle((cl_mem) mask, &ctx),
                    WrapHandle((cl_mem) top_mask, &ctx), channels_,
                    WrapHandle((cl_mem) (size_.gpu_data()), &ctx),
                    WrapHandle((cl_mem) (pooled_size_.gpu_data()), &ctx),
                    WrapHandle((cl_mem) (kernel_shape_.gpu_data()), &ctx),
                    WrapHandle((cl_mem) (ext_kernel_shape_.gpu_data()), &ctx),
                    WrapHandle((cl_mem) (stride_.gpu_data()), &ctx),
                    WrapHandle((cl_mem) (dilation_.gpu_data()), &ctx),
                    WrapHandle((cl_mem) (pad_.gpu_data()), &ctx),
                    WrapHandle((cl_mem) bottom_diff, &ctx)),
                ctx.get_queue());
          }
          break;
          default:
          LOG(FATAL)
          << "Unknown or unsupported pooling method in Backward_gpu().";
        }
      }
#endif  // USE_GREENTEA
    }
  }

INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);

}  // namespace caffe
