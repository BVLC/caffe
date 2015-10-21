#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void im2col_sk_gpu_kernel(const int_tp n, const Dtype* data_im,
                                     const int_tp height, const int_tp width,
                                     const int_tp kernel_h, const int_tp kernel_w,
                                     const int_tp ext_kernel_h,
                                     const int_tp ext_kernel_w, const int_tp pad_h,
                                     const int_tp pad_w, const int_tp stride_h,
                                     const int_tp stride_w, const int_tp kstride_h,
                                     const int_tp kstride_w, const int_tp height_col,
                                     const int_tp width_col, Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int_tp w_out = index % width_col;
    int_tp h_index = index / width_col;
    int_tp h_out = h_index % height_col;
    int_tp channel_in = h_index / height_col;
    int_tp channel_out = channel_in * kernel_h * kernel_w;
    int_tp h_in = h_out * stride_h - pad_h;
    int_tp w_in = w_out * stride_w - pad_w;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int_tp i = 0; i < ext_kernel_h; i += kstride_h) {
      for (int_tp j = 0; j < ext_kernel_w; j += kstride_w) {
        int_tp h = h_in + i;
        int_tp w = w_in + j;
        *data_col_ptr =
            (h >= 0 && w >= 0 && h < height && w < width) ?
                data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template<typename Dtype>
void im2col_sk_gpu(const Dtype* data_im, const int_tp channels, const int_tp height,
                   const int_tp width, const int_tp kernel_h, const int_tp kernel_w,
                   const int_tp pad_h, const int_tp pad_w, const int_tp stride_h,
                   const int_tp stride_w, const int_tp kstride_h, const int_tp kstride_w,
                   Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int_tp ext_kernel_h = (kernel_h - 1) * kstride_h + 1;
  int_tp ext_kernel_w = (kernel_w - 1) * kstride_w + 1;
  int_tp height_col = (height + 2 * pad_h - ext_kernel_h) / stride_h + 1;
  int_tp width_col = (width + 2 * pad_w - ext_kernel_w) / stride_w + 1;
  int_tp num_kernels = channels * height_col * width_col;

  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_sk_gpu_kernel<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(num_kernels),
      CAFFE_CUDA_NUM_THREADS)(
      num_kernels, data_im, height, width, kernel_h, kernel_w,
      ext_kernel_h, ext_kernel_w, pad_h, pad_w,
      stride_h, stride_w, kstride_h, kstride_w,
      height_col, width_col,
      data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_sk_gpu<float>(const float* data_im, const int_tp channels,
                                   const int_tp height, const int_tp width,
                                   const int_tp kernel_h, const int_tp kernel_w,
                                   const int_tp pad_h, const int_tp pad_w,
                                   const int_tp stride_h, const int_tp stride_w,
                                   const int_tp kstride_h, const int_tp kstride_w,
                                   float* data_col);
template void im2col_sk_gpu<double>(const double* data_im, const int_tp channels,
                                    const int_tp height, const int_tp width,
                                    const int_tp kernel_h, const int_tp kernel_w,
                                    const int_tp pad_h, const int_tp pad_w,
                                    const int_tp stride_h, const int_tp stride_w,
                                    const int_tp kstride_h, const int_tp kstride_w,
                                    double* data_col);

template<typename Dtype>
__global__ void im2col_gpu_kernel(const int_tp n, const Dtype* data_im,
                                  const int_tp height, const int_tp width,
                                  const int_tp kernel_h, const int_tp kernel_w,
                                  const int_tp pad_h, const int_tp pad_w,
                                  const int_tp stride_h, const int_tp stride_w,
                                  const int_tp height_col, const int_tp width_col,
                                  Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int_tp w_out = index % width_col;
    int_tp h_index = index / width_col;
    int_tp h_out = h_index % height_col;
    int_tp channel_in = h_index / height_col;
    int_tp channel_out = channel_in * kernel_h * kernel_w;
    int_tp h_in = h_out * stride_h - pad_h;
    int_tp w_in = w_out * stride_w - pad_w;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int_tp i = 0; i < kernel_h; ++i) {
      for (int_tp j = 0; j < kernel_w; ++j) {
        int_tp h = h_in + i;
        int_tp w = w_in + j;
        *data_col_ptr =
            (h >= 0 && w >= 0 && h < height && w < width) ?
                data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template<typename Dtype>
void im2col_gpu(const Dtype* data_im, const int_tp channels, const int_tp height,
                const int_tp width, const int_tp kernel_h, const int_tp kernel_w,
                const int_tp pad_h, const int_tp pad_w, const int_tp stride_h,
                const int_tp stride_w, Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int_tp height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int_tp width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int_tp num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(num_kernels),
      CAFFE_CUDA_NUM_THREADS)(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int_tp channels,
                                const int_tp height, const int_tp width,
                                const int_tp kernel_h, const int_tp kernel_w,
                                const int_tp pad_h, const int_tp pad_w,
                                const int_tp stride_h, const int_tp stride_w,
                                float* data_col);
template void im2col_gpu<double>(const double* data_im, const int_tp channels,
                                 const int_tp height, const int_tp width,
                                 const int_tp kernel_h, const int_tp kernel_w,
                                 const int_tp pad_h, const int_tp pad_w,
                                 const int_tp stride_h, const int_tp stride_w,
                                 double* data_col);

// Support of stride_h and stride_w greater than 1 is not implemented
template<typename Dtype>
__global__ void col2im_sk_gpu_kernel(const int_tp n, const Dtype* data_col,
                                     const int_tp height, const int_tp width,
                                     const int_tp channels, const int_tp patch_h,
                                     const int_tp patch_w, const int_tp ext_patch_h,
                                     const int_tp ext_patch_w, const int_tp pad_h,
                                     const int_tp pad_w, const int_tp stride_h,
                                     const int_tp stride_w, const int_tp kstride_h,
                                     const int_tp kstride_w, const int_tp height_col,
                                     const int_tp width_col, Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    int_tp w = index % width + pad_w;
    int_tp h = (index / width) % height + pad_h;
    int_tp c = index / (width * height);
    // compute the start and end of the output
    int_tp width_col_1 = width_col - 1;
    int_tp height_col_1 = height_col - 1;
    int_tp w_col_start = (w < ext_patch_w) ? w % kstride_w : (w - ext_patch_w) + 1;
    int_tp w_col_end =
        (w >= width_col) ?
            width_col_1 - (width_col_1 - w_col_start) % kstride_w : w;
    int_tp h_col_start = (h < ext_patch_h) ? h % kstride_h : (h - ext_patch_h) + 1;
    int_tp h_col_end =
        (h >= height_col) ?
            height_col_1 - (height_col_1 - h_col_start) % kstride_h : h;
    int_tp w_num = (w - w_col_start) / kstride_w;
    int_tp h_num = (h - h_col_start) / kstride_h;

    int_tp coeff_w_idx = height_col * width_col;
    int_tp coeff_h_idx = patch_w * coeff_w_idx;
    int_tp offset = c * patch_h * coeff_h_idx;
    for (int_tp h_col = h_col_start, h_idx = h_num; h_col <= h_col_end; h_col +=
        kstride_h, --h_idx) {
      for (int_tp w_col = w_col_start, w_idx = w_num; w_col <= w_col_end; w_col +=
          kstride_w, --w_idx) {
        val += data_col[offset + h_idx * coeff_h_idx + w_idx * coeff_w_idx
            + h_col * width_col + w_col];
      }
    }

    data_im[index] = val;
  }
}

template<typename Dtype>
void col2im_sk_gpu(const Dtype* data_col, const int_tp channels, const int_tp height,
                   const int_tp width, const int_tp patch_h, const int_tp patch_w,
                   const int_tp pad_h, const int_tp pad_w, const int_tp stride_h,
                   const int_tp stride_w, const int_tp kstride_h, const int_tp kstride_w,
                   Dtype* data_im) {
  if (stride_w > 1 || stride_h > 1 || pad_h > 0 || pad_w > 0)
    LOG(FATAL)<< "stride greater than 1 or pad greater"
    << " than 0 not tested in col2im_sk_gpu().";
    int_tp ext_patch_h = (patch_h - 1) * kstride_h + 1;
    int_tp ext_patch_w = (patch_w - 1) * kstride_w + 1;
    int_tp height_col = (height + 2 * pad_h - ext_patch_h) / stride_h + 1;
    int_tp width_col = (width + 2 * pad_w - ext_patch_w) / stride_w + 1;
    int_tp num_kernels = channels * height * width;

    col2im_sk_gpu_kernel<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(num_kernels),
        CAFFE_CUDA_NUM_THREADS)(
        num_kernels, data_col, height, width, channels,
        patch_h, patch_w, ext_patch_h, ext_patch_w,
        pad_h, pad_w, stride_h, stride_w, kstride_h, kstride_w,
        height_col, width_col, data_im);
    CUDA_POST_KERNEL_CHECK;
  }

// Explicit instantiation
template void col2im_sk_gpu<float>(const float* data_col, const int_tp channels,
                                   const int_tp height, const int_tp width,
                                   const int_tp patch_h, const int_tp patch_w,
                                   const int_tp pad_h, const int_tp pad_w,
                                   const int_tp stride_h, const int_tp stride_w,
                                   const int_tp kstride_h, const int_tp kstride_w,
                                   float* data_im);
template void col2im_sk_gpu<double>(const double* data_col, const int_tp channels,
                                    const int_tp height, const int_tp width,
                                    const int_tp patch_h, const int_tp patch_w,
                                    const int_tp pad_h, const int_tp pad_w,
                                    const int_tp stride_h, const int_tp stride_w,
                                    const int_tp kstride_h, const int_tp kstride_w,
                                    double* data_im);

template<typename Dtype>
__global__ void col2im_gpu_kernel(const int_tp n, const Dtype* data_col,
                                  const int_tp height, const int_tp width,
                                  const int_tp channels, const int_tp patch_h,
                                  const int_tp patch_w, const int_tp pad_h,
                                  const int_tp pad_w, const int_tp stride_h,
                                  const int_tp stride_w, const int_tp height_col,
                                  const int_tp width_col, Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    int_tp w = index % width + pad_w;
    int_tp h = (index / width) % height + pad_h;
    int_tp c = index / (width * height);

    // compute the start and end of the output
    int_tp w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int_tp w_col_end = min((int_tpc)(w / stride_w + 1), (int_tpc)width_col);
    int_tp h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int_tp h_col_end = min((int_tpc)(h / stride_h + 1), (int_tpc)height_col);

    // equivalent implementation
    int_tp offset = (c * patch_h * patch_w + h * patch_w + w) * height_col
        * width_col;
    int_tp coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
    int_tp coeff_w_col = (1 - stride_w * height_col * width_col);
    for (int_tp h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int_tp w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

template<typename Dtype>
void col2im_gpu(const Dtype* data_col, const int_tp channels, const int_tp height,
                const int_tp width, const int_tp patch_h, const int_tp patch_w,
                const int_tp pad_h, const int_tp pad_w, const int_tp stride_h,
                const int_tp stride_w, Dtype* data_im) {
  int_tp height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int_tp width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int_tp num_kernels = channels * height * width;

  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(num_kernels),
      CAFFE_CUDA_NUM_THREADS)(
      num_kernels, data_col, height, width, channels, patch_h, patch_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int_tp channels,
                                const int_tp height, const int_tp width,
                                const int_tp patch_h, const int_tp patch_w,
                                const int_tp pad_h, const int_tp pad_w,
                                const int_tp stride_h, const int_tp stride_w,
                                float* data_im);
template void col2im_gpu<double>(const double* data_col, const int_tp channels,
                                 const int_tp height, const int_tp width,
                                 const int_tp patch_h, const int_tp patch_w,
                                 const int_tp pad_h, const int_tp pad_w,
                                 const int_tp stride_h, const int_tp stride_w,
                                 double* data_im);

template<typename Dtype>
__global__ void im2col_ndsk_gpu_kernel(const int_tp n, const int_tp num_axes,
                                       const Dtype* data_im,
                                       const int_tp* im_shape,
                                       const int_tp* col_shape,
                                       const int_tp* kernel_shape, const int_tp* pad,
                                       const int_tp* stride, const int_tp* kstride,
                                       Dtype* data_col) {
  int_tp d_temp[6];  // NOLINT(runtime/arrays)
  int_tp d_iter[6];  // NOLINT(runtime/arrays)
  int_tp i;
  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int_tp channel_in = index;
    int_tp channel_out = 1;
    for (i = num_axes - 1; i >= 0; --i) {
      d_temp[i] = channel_in % col_shape[i + 1];
      channel_in /= col_shape[i + 1];
      channel_out *= kernel_shape[i];
    }
    channel_out *= channel_in;
    int_tp data_col_inc = 1;
    for (i = 0; i < num_axes; ++i) {
      channel_out *= col_shape[i + 1];
      channel_out += d_temp[i];
      d_temp[i] = d_temp[i] * stride[i] - pad[i];
      channel_in *= im_shape[i + 1];
      channel_in += d_temp[i];
      data_col_inc *= col_shape[i + 1];
      d_iter[i] = 0;
    }
    Dtype* data_col_ptr = data_col + channel_out;
    const Dtype* data_im_ptr = data_im + channel_in;
    bool incremented;
    do {
      bool in_range = true;
      for (i = 0; i < num_axes; ++i) {
        const int_tp d_iter_im = d_iter[i] + d_temp[i];
        in_range &= d_iter_im >= 0 && d_iter_im < im_shape[i + 1];
        if (!in_range) {
          break;
        }
      }

      // Write column data
      if (in_range) {
        int_tp data_im_offset = d_iter[0];
        for (i = 1; i < num_axes; ++i) {
          data_im_offset *= im_shape[i + 1];
          data_im_offset += d_iter[i];
        }
        *data_col_ptr = data_im_ptr[data_im_offset];
      } else {
        *data_col_ptr = 0;
      }

      data_col_ptr += data_col_inc;
      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        // Old: const int_tp d_max = kernel_shape[i];
        // New (strided, limit is the external kernel size):
        const int_tp d_max = (kernel_shape[i] - 1) * kstride[i] + 1;
        if (d_iter[i] > d_max - kstride[i]) {
          d_iter[i] = 0;
        } else {  // d_iter[i] <= d_max - kstride[i]
          // Old: ++d_iter[i];
          // New (strided, increment by the stride each time):
          d_iter[i] += kstride[i];
          incremented = true;
          break;
        }
      }  // for (int_tp i = num_axes - 1; i >= 0; --i)
    } while (incremented);  // do
  }  // CUDA_KERNEL_LOOP(index, n)
}

template<typename Dtype>
__global__ void col2im_ndsk_gpu_kernel(const int_tp n, const int_tp num_axes,
                                       const Dtype* data_col,
                                       const int_tp* im_shape,
                                       const int_tp* col_shape,
                                       const int_tp* kernel_shape, const int_tp* pad,
                                       const int_tp* stride, const int_tp* kstride,
                                       Dtype* data_im) {
  int_tp d_im[6];  // NOLINT(runtime/arrays)
  int_tp d_col_size[6];  // NOLINT(runtime/arrays)
  int_tp d_col_iter[6];  // NOLINT(runtime/arrays)
  int_tp d_col_start[6];  // NOLINT(runtime/arrays)
  int_tp d_col_end[6];  // NOLINT(runtime/arrays)
  int_tp d_ext_patch[6];  // NOLINT(runtime/arrays)
  int_tp d_idx[6];  // NOLINT(runtime/arrays)

  for (int_tp i = num_axes - 1; i >= 0; --i) {
    d_ext_patch[i] = (kernel_shape[i] - 1) * kstride[i] + 1;
    d_col_size[i] = (im_shape[i + 1] + 2 * pad[i] - d_ext_patch[i]) / stride[i]
        + 1;
  }

  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int_tp channel_im = index;
    // Calculate d_im (image dimensions).
    for (int_tp i = num_axes - 1; i >= 0; --i) {
      d_im[i] = channel_im % im_shape[i + 1] + pad[i];
      channel_im /= im_shape[i + 1];
    }
    // Calculate col start/end indices.
    bool done = false;
    for (int_tp i = 0; i < num_axes; ++i) {
      // Old:
      /*d_col_start[i] = d_col_iter[i] =
       (d_im[i] < kernel_shape[i]) ?
       0 : (d_im[i] - kernel_shape[i]) / stride[i] + 1;
       d_col_end[i] = min(d_im[i] / stride[i] + 1, col_shape[i + 1]);*/
      // New:
      d_col_start[i] =
          (d_im[i] < d_ext_patch[i]) ?
              d_im[i] % kstride[i] : (d_im[i] - d_ext_patch[i]) + 1;
      d_col_iter[i] = d_col_start[i];
      d_idx[i] = (d_im[i] - d_col_start[i]) / kstride[i];
      d_col_end[i] =
          (d_im[i] >= d_col_size[i]) ?
              (d_col_size[i] - 1)
                  - ((d_col_size[i] - 1) - d_col_start[i]) % kstride[i] :
              d_im[i];
      if (d_col_start[i] > d_col_end[i]) {
        // Skip computation if the dimension is 0 at any spatial axis --
        // final val will be 0.
        data_im[index] = 0;
        done = true;
        break;  // for (int_tp i = 0; i < num_axes; ++i)
      }
    }
    if (done) {
      continue;  // CUDA_KERNEL_LOOP(index, n)
    }
    // Loop over the col to compute the output val.
    Dtype val = 0;
    bool incremented = true;
    do {
      // Compute the final offset.
      int_tp final_offset = 0;
      int_tp coeff_prod = 1;
      for (int_tp i = num_axes - 1; i >= 0; --i) {
        final_offset += d_col_iter[i] * coeff_prod;
        coeff_prod *= d_col_size[i];
      }
      for (int_tp i = num_axes - 1; i >= 0; --i) {
        final_offset += d_idx[i] * coeff_prod;
        coeff_prod *= kernel_shape[i];
      }
      final_offset += channel_im * coeff_prod;
      val += data_col[final_offset];
      incremented = false;
      for (int_tp i = num_axes - 1; i >= 0; --i) {
        if (d_col_iter[i] > d_col_end[i] - kstride[i]) {
          d_col_iter[i] = d_col_start[i];
          d_idx[i] = (d_im[i] - d_col_start[i]) / kstride[i];
        } else {  // d_col_iter[i] <= d_max - kstride[1]
          d_col_iter[i] += kstride[i];
          --d_idx[i];
          incremented = true;
          break;  // for (int_tp i = num_axes - 1; i >= 0; --i)
        }
      }  // for (int_tp i = num_axes - 1; i >= 0; --i)
    } while (incremented);
    data_im[index] = val;
  }  // CUDA_KERNEL_LOOP(index, n)
}

template<typename Dtype>
void im2col_ndsk_gpu(const Dtype* data_im, const int_tp num_spatial_axes,
                     const int_tp num_kernels, const int_tp* im_shape,
                     const int_tp* col_shape, const int_tp* kernel_shape,
                     const int_tp* pad, const int_tp* stride, const int_tp* kstride,
                     Dtype* data_col) {
  im2col_ndsk_gpu_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
  CUDA_KERNEL(CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS) (
      num_kernels, num_spatial_axes, data_im, im_shape, col_shape,
      kernel_shape, pad, stride, kstride, data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_ndsk_gpu(const float* data_im, const int_tp num_spatial_axes,
                              const int_tp num_kernels, const int_tp* im_shape,
                              const int_tp* col_shape, const int_tp* kernel_shape,
                              const int_tp* pad, const int_tp* stride,
                              const int_tp* kstride, float* data_col);
template void im2col_ndsk_gpu(const double* data_im, const int_tp num_spatial_axes,
                              const int_tp num_kernels, const int_tp* im_shape,
                              const int_tp* col_shape, const int_tp* kernel_shape,
                              const int_tp* pad, const int_tp* stride,
                              const int_tp* kstride, double* data_col);

template<typename Dtype>
void col2im_ndsk_gpu(const Dtype* data_col, const int_tp num_spatial_axes,
                     const int_tp im_size, const int_tp* im_shape,
                     const int_tp* col_shape, const int_tp* kernel_shape,
                     const int_tp* pad, const int_tp* stride, const int_tp* kstride,
                     Dtype* data_im) {
  col2im_ndsk_gpu_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
  CUDA_KERNEL(CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS)(
      im_size, num_spatial_axes, data_col, im_shape, col_shape,
      kernel_shape, pad, stride, kstride, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_ndsk_gpu(const float* data_col, const int_tp num_axes,
                              const int_tp im_size, const int_tp* im_shape,
                              const int_tp* col_shape, const int_tp* kernel_shape,
                              const int_tp* pad, const int_tp* stride,
                              const int_tp* kstride, float* data_im);
template void col2im_ndsk_gpu(const double* data_col, const int_tp num_axes,
                              const int_tp im_size, const int_tp* im_shape,
                              const int_tp* col_shape, const int_tp* kernel_shape,
                              const int_tp* pad, const int_tp* stride,
                              const int_tp* kstride, double* data_im);

template<typename Dtype>
__global__ void im2col_nd_gpu_kernel(const int_tp n, const int_tp num_axes,
                                     const Dtype* data_im, const int_tp* im_shape,
                                     const int_tp* col_shape,
                                     const int_tp* kernel_shape, const int_tp* pad,
                                     const int_tp* stride, Dtype* data_col) {
  int_tp d_temp[6];  // NOLINT(runtime/arrays)
  int_tp d_iter[6];  // NOLINT(runtime/arrays)
  int_tp i;
  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int_tp channel_in = index;
    int_tp channel_out = 1;
    for (i = num_axes - 1; i >= 0; --i) {
      d_temp[i] = channel_in % col_shape[i + 1];
      channel_in /= col_shape[i + 1];
      channel_out *= kernel_shape[i];
    }
    channel_out *= channel_in;
    int_tp data_col_inc = 1;
    for (i = 0; i < num_axes; ++i) {
      channel_out *= col_shape[i + 1];
      channel_out += d_temp[i];
      d_temp[i] = d_temp[i] * stride[i] - pad[i];
      channel_in *= im_shape[i + 1];
      channel_in += d_temp[i];
      data_col_inc *= col_shape[i + 1];
      d_iter[i] = 0;
    }
    Dtype* data_col_ptr = data_col + channel_out;
    const Dtype* data_im_ptr = data_im + channel_in;
    bool incremented;
    do {
      bool in_range = true;
      for (i = 0; i < num_axes; ++i) {
        const int_tp d_iter_im = d_iter[i] + d_temp[i];
        in_range &= d_iter_im >= 0 && d_iter_im < im_shape[i + 1];
        if (!in_range) {
          break;
        }
      }
      if (in_range) {
        int_tp data_im_offset = d_iter[0];
        for (i = 1; i < num_axes; ++i) {
          data_im_offset *= im_shape[i + 1];
          data_im_offset += d_iter[i];
        }
        *data_col_ptr = data_im_ptr[data_im_offset];
      } else {
        *data_col_ptr = 0;
      }
      data_col_ptr += data_col_inc;
      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        const int_tp d_max = kernel_shape[i];
        if (d_iter[i] == d_max - 1) {
          d_iter[i] = 0;
        } else {  // d_iter[i] < d_max - 1
          ++d_iter[i];
          incremented = true;
          break;
        }
      }  // for (int_tp i = num_axes - 1; i >= 0; --i)
    } while (incremented);  // do
  }  // CUDA_KERNEL_LOOP(index, n)
}

template<typename Dtype>
void im2col_nd_gpu(const Dtype* data_im, const int_tp num_spatial_axes,
                   const int_tp num_kernels, const int_tp* im_shape,
                   const int_tp* col_shape, const int_tp* kernel_shape,
                   const int_tp* pad, const int_tp* stride, Dtype* data_col) {
  im2col_nd_gpu_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
  CUDA_KERNEL(CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS)(
      num_kernels, num_spatial_axes, data_im, im_shape, col_shape,
      kernel_shape, pad, stride, data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_nd_gpu<float>(const float* data_im,
                                   const int_tp num_spatial_axes,
                                   const int_tp col_size, const int_tp* im_shape,
                                   const int_tp* col_shape,
                                   const int_tp* kernel_shape, const int_tp* pad,
                                   const int_tp* stride, float* data_col);
template void im2col_nd_gpu<double>(const double* data_im,
                                    const int_tp num_spatial_axes,
                                    const int_tp col_size, const int_tp* im_shape,
                                    const int_tp* col_shape,
                                    const int_tp* kernel_shape, const int_tp* pad,
                                    const int_tp* stride, double* data_col);

template<typename Dtype>
__global__ void col2im_nd_gpu_kernel(const int_tp n, const int_tp num_axes,
                                     const Dtype* data_col, const int_tp* im_shape,
                                     const int_tp* col_shape,
                                     const int_tp* kernel_shape, const int_tp* pad,
                                     const int_tp* stride, Dtype* data_im) {
  int_tp d_im[6];  // NOLINT(runtime/arrays)
  int_tp d_col_iter[6];  // NOLINT(runtime/arrays)
  int_tp d_col_start[6];  // NOLINT(runtime/arrays)
  int_tp d_col_end[6];  // NOLINT(runtime/arrays)
  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int_tp channel_im = index;
    // Calculate d_im (image dimensions).
    for (int_tp i = num_axes - 1; i >= 0; --i) {
      d_im[i] = channel_im % im_shape[i + 1] + pad[i];
      channel_im /= im_shape[i + 1];
    }
    // Calculate col start/end indices.
    bool done = false;
    for (int_tp i = 0; i < num_axes; ++i) {
      d_col_start[i] = d_col_iter[i] =
          (d_im[i] < kernel_shape[i]) ?
              0 : (d_im[i] - kernel_shape[i]) / stride[i] + 1;
      d_col_end[i] = min((int_tpc)(d_im[i] / stride[i] + 1),
                         (int_tpc)(col_shape[i + 1]));
      if (d_col_start[i] >= d_col_end[i]) {
        // Skip computation if the dimension is 0 at any spatial axis --
        // final val will be 0.
        data_im[index] = 0;
        done = true;
        break;  // for (int_tp i = 0; i < num_axes; ++i)
      }
    }
    if (done) {
      continue;  // CUDA_KERNEL_LOOP(index, n)
    }
    // Loop over the col to compute the output val.
    Dtype val = 0;
    bool incremented = true;
    do {
      // Compute the final offset.
      int_tp final_offset = 0;
      int_tp kernel_shape_prod = 1;
      for (int_tp i = num_axes - 1; i >= 0; --i) {
        final_offset += (d_im[i] - d_col_iter[i] * stride[i])
            * kernel_shape_prod;
        kernel_shape_prod *= kernel_shape[i];
      }
      final_offset += kernel_shape_prod * channel_im;
      for (int_tp i = 0; i < num_axes; ++i) {
        final_offset *= col_shape[i + 1];
        final_offset += d_col_iter[i];
      }
      val += data_col[final_offset];
      incremented = false;
      for (int_tp i = num_axes - 1; i >= 0; --i) {
        const int_tp d_max = d_col_end[i];
        if (d_col_iter[i] == d_max - 1) {
          d_col_iter[i] = d_col_start[i];
        } else {  // d_col_iter[i] < d_max - 1
          ++d_col_iter[i];
          incremented = true;
          break;  // for (int_tp i = num_axes - 1; i >= 0; --i)
        }
      }  // for (int_tp i = num_axes - 1; i >= 0; --i)
    } while (incremented);
    data_im[index] = val;
  }  // CUDA_KERNEL_LOOP(index, n)
}

template<typename Dtype>
void col2im_nd_gpu(const Dtype* data_col, const int_tp num_spatial_axes,
                   const int_tp im_size, const int_tp* im_shape, const int_tp* col_shape,
                   const int_tp* kernel_shape, const int_tp* pad, const int_tp* stride,
                   Dtype* data_im) {
  col2im_nd_gpu_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
  CUDA_KERNEL(CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS)(
      im_size, num_spatial_axes, data_col, im_shape, col_shape,
      kernel_shape, pad, stride, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_nd_gpu<float>(const float* data_col,
                                   const int_tp num_spatial_axes,
                                   const int_tp im_size, const int_tp* im_shape,
                                   const int_tp* col_shape,
                                   const int_tp* kernel_shape, const int_tp* pad,
                                   const int_tp* stride, float* data_im);
template void col2im_nd_gpu<double>(const double* data_col,
                                    const int_tp num_spatial_axes,
                                    const int_tp im_size, const int_tp* im_shape,
                                    const int_tp* col_shape,
                                    const int_tp* kernel_shape, const int_tp* pad,
                                    const int_tp* stride, double* data_im);

#endif  // USE_CUDA
}  // namespace caffe
