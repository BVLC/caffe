#ifdef USE_OCL
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"

extern "C" const char _cl_im2col_start;
extern "C" const char _cl_im2col_end;

namespace caffe {

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;

  ClState& clState = Caffe::cl_state();

  ClMemOff<Dtype> buf_data_im = clState.get_buffer_mem(data_im);
  ClMemOff<Dtype> buf_data_col = clState.get_buffer_mem(data_col);

  int offset_data_im = buf_data_im.offset;
  int offset_data_col = buf_data_col.offset;

  uint mem_base_address_align =
      (clState.get_properties().mem_base_addr_align/8) / sizeof(Dtype);

  size_t aligned_offset_data_im =
      static_cast<int>(offset_data_im /
      (static_cast<float>(mem_base_address_align))) * mem_base_address_align;
  cl_int offset_offset_data_im =
      ((offset_data_im / (static_cast<float>(mem_base_address_align))) -
      aligned_offset_data_im / mem_base_address_align) * mem_base_address_align;
  cl_mem bufDataIm = clState.create_subbuffer(data_im, aligned_offset_data_im);

  size_t aligned_offset_data_col =
      (static_cast<int>(offset_data_col /
      (static_cast<float>(mem_base_address_align)))) * mem_base_address_align;
  cl_int offset_offset_data_col =
      ((offset_data_col / (static_cast<float>(mem_base_address_align))) -
      aligned_offset_data_col / mem_base_address_align) *
      mem_base_address_align;
  cl_mem bufDataCol =
      clState.create_subbuffer(data_col, aligned_offset_data_col);

  cl_uint argIdx = 0;
  clState.submit_program("im2col", &_cl_im2col_start, &_cl_im2col_end);
  ClKernel kernel = clState.get_kernel("im2col_gpu");
  kernel.set_arg(argIdx++, num_kernels);
  kernel.set_arg_mem(argIdx++, bufDataIm);
  kernel.set_arg(argIdx++, offset_offset_data_im);
  kernel.set_arg(argIdx++, height);
  kernel.set_arg(argIdx++, width);
  kernel.set_arg(argIdx++, kernel_h);
  kernel.set_arg(argIdx++, kernel_w);
  kernel.set_arg(argIdx++, pad_h);
  kernel.set_arg(argIdx++, pad_w);
  kernel.set_arg(argIdx++, stride_h);
  kernel.set_arg(argIdx++, stride_w);
  kernel.set_arg(argIdx++, height_col);
  kernel.set_arg(argIdx++, width_col);
  kernel.set_arg(argIdx++, offset_offset_data_col);
  kernel.set_arg_mem(argIdx++, bufDataCol);
  kernel.enqueue(num_kernels);

  clReleaseMemObject(bufDataIm);
  clReleaseMemObject(bufDataCol);
}

// Explicit instantiation
template void im2col_gpu<float>(const float* data_im,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col);
template void im2col_gpu<double>(const double* data_im,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_col);

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    Dtype* data_im) {
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  ClState& clState = Caffe::cl_state();
  uint mem_base_address_align =
      (clState.get_properties().mem_base_addr_align / 8) / sizeof(Dtype);

  ClMemOff<Dtype> buf_data_col = clState.get_buffer_mem(data_col);
  ClMemOff<Dtype> buf_data_im = clState.get_buffer_mem(data_im);

  int offset_data_col = buf_data_col.offset;
  int offset_data_im = buf_data_im.offset;

  size_t aligned_offset_data_im =
      static_cast<int>(offset_data_im /
      (static_cast<float>(mem_base_address_align))) * mem_base_address_align;
  cl_int offset_offset_data_im =
      ((offset_data_im/(static_cast<float>(mem_base_address_align))) -
      aligned_offset_data_im / mem_base_address_align) * mem_base_address_align;
  cl_mem bufDataIm = clState.create_subbuffer(data_im, aligned_offset_data_im);

  size_t aligned_offset_data_col =
      (static_cast<int>(offset_data_col /
      (static_cast<float>(mem_base_address_align)))) * mem_base_address_align;
  cl_int offset_offset_data_col =
      ((offset_data_col / (static_cast<float>(mem_base_address_align))) -
      aligned_offset_data_col / mem_base_address_align) *
      mem_base_address_align;
  cl_mem bufDataCol =
      clState.create_subbuffer(data_col, aligned_offset_data_col);

  cl_uint argIdx = 0;
  clState.submit_program("im2col", &_cl_im2col_start, &_cl_im2col_end);
  ClKernel kernel = clState.get_kernel("col2im_gpu");
  kernel.set_arg(argIdx++, num_kernels);
  kernel.set_arg_mem(argIdx++, bufDataCol);
  kernel.set_arg(argIdx++, offset_offset_data_col);
  kernel.set_arg(argIdx++, height);
  kernel.set_arg(argIdx++, width);
  kernel.set_arg(argIdx++, channels);
  kernel.set_arg(argIdx++, patch_h);
  kernel.set_arg(argIdx++, patch_w);
  kernel.set_arg(argIdx++, pad_h);
  kernel.set_arg(argIdx++, pad_w);
  kernel.set_arg(argIdx++, stride_h);
  kernel.set_arg(argIdx++, stride_w);
  kernel.set_arg(argIdx++, height_col);
  kernel.set_arg(argIdx++, width_col);
  kernel.set_arg(argIdx++, offset_offset_data_im);
  kernel.set_arg_mem(argIdx++, bufDataIm);
  kernel.enqueue(num_kernels);

  clReleaseMemObject(bufDataCol);
  clReleaseMemObject(bufDataIm);
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col,
    const int channels, const int height, const int width, const int patch_h,
    const int patch_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im);
template void col2im_gpu<double>(const double* data_col,
    const int channels, const int height, const int width, const int patch_h,
    const int patch_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im);

}  // namespace caffe

#endif  // USE_OCL
