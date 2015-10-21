/*
 * greentea_im2col.cpp
 *
 *  Created on: Apr 8, 2015
 *      Author: Fabian Tschopp
 */
#include "caffe/common.hpp"
#ifdef USE_GREENTEA
#include "caffe/greentea/greentea_im2col.hpp"

namespace caffe {

template<typename Dtype>
void greentea_im2col_sk_gpu(viennacl::ocl::program *prog,
                            viennacl::ocl::context *ctx, const cl_mem data_im,
                            const int_tp data_offset, const int_tp channels,
                            const int_tp height, const int_tp width,
                            const int_tp kernel_h, const int_tp kernel_w,
                            const int_tp pad_h, const int_tp pad_w,
                            const int_tp stride_h, const int_tp stride_w,
                            const int_tp kstride_h, const int_tp kstride_w,
                            cl_mem data_col) {
  int_tp ext_kernel_h = (kernel_h - 1) * kstride_h + 1;
  int_tp ext_kernel_w = (kernel_w - 1) * kstride_w + 1;
  int_tp height_col = (height + 2 * pad_h - ext_kernel_h) / stride_h + 1;
  int_tp width_col = (width + 2 * pad_w - ext_kernel_w) / stride_w + 1;
  int_tp num_kernels = channels * height_col * width_col;

  viennacl::ocl::kernel &kernel = prog->get_kernel(
      CL_KERNEL_SELECT("im2col_sk"));

  viennacl::ocl::enqueue(
      kernel(num_kernels, WrapHandle(data_im, ctx), data_offset, height, width,
             kernel_h, kernel_w, ext_kernel_h, ext_kernel_w, pad_h, pad_w,
             stride_h, stride_w, kstride_h, kstride_w, height_col, width_col,
             WrapHandle(data_col, ctx)),
      ctx->get_queue());
}

// Explicit instantiation
template void greentea_im2col_sk_gpu<float>(viennacl::ocl::program *prog,
                                            viennacl::ocl::context *ctx,
                                            const cl_mem data_im,
                                            const int_tp data_offset,
                                            const int_tp channels,
                                            const int_tp height, const int_tp width,
                                            const int_tp kernel_h,
                                            const int_tp kernel_w, const int_tp pad_h,
                                            const int_tp pad_w, const int_tp stride_h,
                                            const int_tp stride_w,
                                            const int_tp kstride_h,
                                            const int_tp kstride_w,
                                            cl_mem data_col);

template void greentea_im2col_sk_gpu<double>(viennacl::ocl::program *prog,
                                             viennacl::ocl::context *ctx,
                                             const cl_mem data_im,
                                             const int_tp data_offset,
                                             const int_tp channels,
                                             const int_tp height, const int_tp width,
                                             const int_tp kernel_h,
                                             const int_tp kernel_w,
                                             const int_tp pad_h, const int_tp pad_w,
                                             const int_tp stride_h,
                                             const int_tp stride_w,
                                             const int_tp kstride_h,
                                             const int_tp kstride_w,
                                             cl_mem data_col);

template<typename Dtype>
void greentea_col2im_sk_gpu(viennacl::ocl::program *prog,
                            viennacl::ocl::context *ctx, const cl_mem data_col,
                            const int_tp channels, const int_tp height,
                            const int_tp width, const int_tp patch_h,
                            const int_tp patch_w, const int_tp pad_h, const int_tp pad_w,
                            const int_tp stride_h, const int_tp stride_w,
                            const int_tp kstride_h, const int_tp kstride_w,
                            cl_mem data_im, const int_tp data_offset) {
  if (stride_w > 1 || stride_h > 1 || pad_h > 0 || pad_w > 0) {
    LOG(FATAL)<< "stride greater than 1 or pad greater than 0"
    << " not tested in col2im_sk_gpu().";
  }

  int_tp ext_patch_h = (patch_h - 1) * kstride_h + 1;
  int_tp ext_patch_w = (patch_w - 1) * kstride_w + 1;
  int_tp height_col = (height + 2 * pad_h - ext_patch_h) / stride_h + 1;
  int_tp width_col = (width + 2 * pad_w - ext_patch_w) / stride_w + 1;
  int_tp num_kernels = channels * height * width;

  viennacl::ocl::kernel &kernel = prog->get_kernel(
      CL_KERNEL_SELECT("col2im_sk"));

  viennacl::ocl::enqueue(
      kernel(num_kernels, WrapHandle(data_col, ctx), height, width, channels,
          patch_h, patch_w, ext_patch_h, ext_patch_w,
          pad_h, pad_w, stride_h, stride_w, kstride_h, kstride_w,
          height_col, width_col, WrapHandle(data_im, ctx), data_offset),
      ctx->get_queue());
}

template void greentea_col2im_sk_gpu<float>(viennacl::ocl::program *prog,
                                            viennacl::ocl::context *ctx,
                                            const cl_mem data_col,
                                            const int_tp channels,
                                            const int_tp height, const int_tp width,
                                            const int_tp patch_h,
                                            const int_tp patch_w, const int_tp pad_h,
                                            const int_tp pad_w, const int_tp stride_h,
                                            const int_tp stride_w,
                                            const int_tp kstride_h,
                                            const int_tp kstride_w, cl_mem data_im,
                                            const int_tp data_offset);

template void greentea_col2im_sk_gpu<double>(viennacl::ocl::program *prog,
                                             viennacl::ocl::context *ctx,
                                             const cl_mem data_col,
                                             const int_tp channels,
                                             const int_tp height, const int_tp width,
                                             const int_tp patch_h,
                                             const int_tp patch_w, const int_tp pad_h,
                                             const int_tp pad_w,
                                             const int_tp stride_h,
                                             const int_tp stride_w,
                                             const int_tp kstride_h,
                                             const int_tp kstride_w,
                                             cl_mem data_im,
                                             const int_tp data_offset);

template<typename Dtype>
void greentea_im2col_gpu(viennacl::ocl::program *prog,
                         viennacl::ocl::context *ctx, const cl_mem data_im,
                         const int_tp data_im_off, const int_tp channels,
                         const int_tp height, const int_tp width, const int_tp kernel_h,
                         const int_tp kernel_w, const int_tp pad_h, const int_tp pad_w,
                         const int_tp stride_h, const int_tp stride_w,
                         cl_mem data_col, const int_tp data_col_off) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int_tp height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int_tp width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int_tp num_kernels = channels * height_col * width_col;

  viennacl::ocl::kernel &kernel = prog->get_kernel(CL_KERNEL_SELECT("im2col"));

  viennacl::ocl::enqueue(
      kernel(num_kernels, WrapHandle(data_im, ctx), data_im_off, height, width,
             kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, height_col,
             width_col, WrapHandle(data_col, ctx), data_col_off),
      ctx->get_queue());
}

template void greentea_im2col_gpu<float>(viennacl::ocl::program *prog,
                                         viennacl::ocl::context *ctx,
                                         const cl_mem data_im,
                                         const int_tp data_im_off,
                                         const int_tp channels, const int_tp height,
                                         const int_tp width, const int_tp kernel_h,
                                         const int_tp kernel_w, const int_tp pad_h,
                                         const int_tp pad_w, const int_tp stride_h,
                                         const int_tp stride_w, cl_mem data_col,
                                         const int_tp data_col_off);

template void greentea_im2col_gpu<double>(viennacl::ocl::program *prog,
                                          viennacl::ocl::context *ctx,
                                          const cl_mem data_im,
                                          const int_tp data_im_off,
                                          const int_tp channels, const int_tp height,
                                          const int_tp width, const int_tp kernel_h,
                                          const int_tp kernel_w, const int_tp pad_h,
                                          const int_tp pad_w, const int_tp stride_h,
                                          const int_tp stride_w, cl_mem data_col,
                                          const int_tp data_col_off);

template<typename Dtype>
void greentea_col2im_gpu(viennacl::ocl::program *prog,
                         viennacl::ocl::context *ctx, const cl_mem data_col,
                         const int_tp data_col_off, const int_tp channels,
                         const int_tp height, const int_tp width, const int_tp patch_h,
                         const int_tp patch_w, const int_tp pad_h, const int_tp pad_w,
                         const int_tp stride_h, const int_tp stride_w, cl_mem data_im,
                         const int_tp data_im_off) {
  int_tp height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int_tp width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int_tp num_kernels = channels * height * width;

  viennacl::ocl::kernel &kernel = prog->get_kernel(CL_KERNEL_SELECT("col2im"));

  viennacl::ocl::enqueue(
      kernel(num_kernels, WrapHandle(data_col, ctx), data_col_off, height,
             width, channels, patch_h, patch_w, pad_h, pad_w, stride_h,
             stride_w, height_col, width_col, WrapHandle(data_im, ctx),
             data_im_off),
      ctx->get_queue());
}

template void greentea_col2im_gpu<float>(viennacl::ocl::program *prog,
                                         viennacl::ocl::context *ctx,
                                         const cl_mem data_col,
                                         const int_tp data_col_off,
                                         const int_tp channels, const int_tp height,
                                         const int_tp width, const int_tp patch_h,
                                         const int_tp patch_w, const int_tp pad_h,
                                         const int_tp pad_w, const int_tp stride_h,
                                         const int_tp stride_w, cl_mem data_im,
                                         const int_tp data_im_off);
template void greentea_col2im_gpu<double>(viennacl::ocl::program *prog,
                                          viennacl::ocl::context *ctx,
                                          const cl_mem data_col,
                                          const int_tp data_col_off,
                                          const int_tp channels, const int_tp height,
                                          const int_tp width, const int_tp patch_h,
                                          const int_tp patch_w, const int_tp pad_h,
                                          const int_tp pad_w, const int_tp stride_h,
                                          const int_tp stride_w, cl_mem data_im,
                                          const int_tp data_im_off);

template<typename Dtype>
void greentea_im2col_nd_gpu(viennacl::ocl::program *prog,
                            viennacl::ocl::context *ctx, cl_mem data_im,
                            const int_tp data_off, const int_tp num_spatial_axes,
                            const int_tp channel_axis,
                            const int_tp num_kernels,
                            cl_mem im_shape, cl_mem col_shape,
                            cl_mem kernel_shape, cl_mem pad, cl_mem stride,
                            cl_mem data_col, int_tp data_col_off) {
  viennacl::ocl::kernel &kernel = prog->get_kernel(
      CL_KERNEL_SELECT("im2col_nd"));

  viennacl::ocl::enqueue(
      kernel(num_kernels, num_spatial_axes, channel_axis,
             WrapHandle(data_im, ctx), data_off, WrapHandle(im_shape, ctx),
             WrapHandle(col_shape, ctx), WrapHandle(kernel_shape, ctx),
             WrapHandle(pad, ctx), WrapHandle(stride, ctx),
             WrapHandle(data_col, ctx), data_col_off),
      ctx->get_queue());
}

// Explicit instantiation
template void greentea_im2col_nd_gpu<float>(viennacl::ocl::program *prog,
                                            viennacl::ocl::context *ctx,
                                            cl_mem data_im, const int_tp data_off,
                                            const int_tp num_spatial_axes,
                                            const int_tp channel_axis,
                                            const int_tp num_kernels,
                                            cl_mem im_shape, cl_mem col_shape,
                                            cl_mem kernel_shape, cl_mem pad,
                                            cl_mem stride, cl_mem data_col,
                                            int_tp data_col_off);

template void greentea_im2col_nd_gpu<double>(viennacl::ocl::program *prog,
                                             viennacl::ocl::context *ctx,
                                             cl_mem data_im, const int_tp data_off,
                                             const int_tp num_spatial_axes,
                                             const int_tp channel_axis,
                                             const int_tp num_kernels,
                                             cl_mem im_shape, cl_mem col_shape,
                                             cl_mem kernel_shape, cl_mem pad,
                                             cl_mem stride, cl_mem data_col,
                                             int_tp data_col_off);

template<typename Dtype>
void greentea_col2im_nd_gpu(viennacl::ocl::program *prog,
                            viennacl::ocl::context *ctx, cl_mem data_col,
                            const int_tp data_col_off, const int_tp num_spatial_axes,
                            const int_tp channel_axis, const int_tp im_size,
                            cl_mem im_shape, cl_mem col_shape,
                            cl_mem kernel_shape, cl_mem pad, cl_mem stride,
                            cl_mem data_im, int_tp data_off) {
  viennacl::ocl::kernel &kernel = prog->get_kernel(
      CL_KERNEL_SELECT("col2im_nd"));

  viennacl::ocl::enqueue(
      kernel(im_size, num_spatial_axes, channel_axis, WrapHandle(data_col, ctx),
             data_col_off, WrapHandle(im_shape, ctx),
             WrapHandle(col_shape, ctx), WrapHandle(kernel_shape, ctx),
             WrapHandle(pad, ctx), WrapHandle(stride, ctx),
             WrapHandle(data_im, ctx), data_off),
      ctx->get_queue());
}

// Explicit instantiation
template void greentea_col2im_nd_gpu<float>(viennacl::ocl::program *prog,
                                            viennacl::ocl::context *ctx,
                                            cl_mem data_col,
                                            const int_tp data_col_off,
                                            const int_tp num_spatial_axes,
                                            const int_tp channel_axis,
                                            const int_tp im_size, cl_mem im_shape,
                                            cl_mem col_shape,
                                            cl_mem kernel_shape, cl_mem pad,
                                            cl_mem stride, cl_mem data_im,
                                            int_tp data_off);

template void greentea_col2im_nd_gpu<double>(viennacl::ocl::program *prog,
                                             viennacl::ocl::context *ctx,
                                             cl_mem data_col,
                                             const int_tp data_col_off,
                                             const int_tp num_spatial_axes,
                                             const int_tp channel_axis,
                                             const int_tp im_size, cl_mem im_shape,
                                             cl_mem col_shape,
                                             cl_mem kernel_shape, cl_mem pad,
                                             cl_mem stride, cl_mem data_im,
                                             int_tp data_off);

template<typename Dtype>
void greentea_im2col_ndsk_gpu(viennacl::ocl::program *prog,
                              viennacl::ocl::context *ctx, cl_mem data_im,
                              const int_tp data_off, const int_tp num_spatial_axes,
                              const int_tp num_kernels, cl_mem im_shape,
                              cl_mem col_shape, cl_mem kernel_shape, cl_mem pad,
                              cl_mem stride, cl_mem kstride, cl_mem data_col,
                              int_tp data_col_off) {
  viennacl::ocl::kernel &kernel = prog->get_kernel(
      CL_KERNEL_SELECT("im2col_ndsk"));

  viennacl::ocl::enqueue(
      kernel(num_kernels, num_spatial_axes, WrapHandle(data_im, ctx), data_off,
             WrapHandle(im_shape, ctx), WrapHandle(col_shape, ctx),
             WrapHandle(kernel_shape, ctx), WrapHandle(pad, ctx),
             WrapHandle(stride, ctx), WrapHandle(kstride, ctx),
             WrapHandle(data_col, ctx), data_col_off),
      ctx->get_queue());
}

// Explicit instantiation
template void greentea_im2col_ndsk_gpu<float>(viennacl::ocl::program *prog,
                                              viennacl::ocl::context *ctx,
                                              cl_mem data_im,
                                              const int_tp data_off,
                                              const int_tp num_spatial_axes,
                                              const int_tp num_kernels,
                                              cl_mem im_shape, cl_mem col_shape,
                                              cl_mem kernel_shape, cl_mem pad,
                                              cl_mem stride, cl_mem kstride,
                                              cl_mem data_col,
                                              int_tp data_col_off);

template void greentea_im2col_ndsk_gpu<double>(viennacl::ocl::program *prog,
                                               viennacl::ocl::context *ctx,
                                               cl_mem data_im,
                                               const int_tp data_off,
                                               const int_tp num_spatial_axes,
                                               const int_tp num_kernels,
                                               cl_mem im_shape,
                                               cl_mem col_shape,
                                               cl_mem kernel_shape, cl_mem pad,
                                               cl_mem stride, cl_mem kstride,
                                               cl_mem data_col,
                                               int_tp data_col_off);

template<typename Dtype>
void greentea_col2im_ndsk_gpu(viennacl::ocl::program *prog,
                              viennacl::ocl::context *ctx, cl_mem data_col,
                              const int_tp data_col_off,
                              const int_tp num_spatial_axes, const int_tp im_size,
                              cl_mem im_shape, cl_mem col_shape,
                              cl_mem kernel_shape, cl_mem pad, cl_mem stride,
                              cl_mem kstride, cl_mem data_im, int_tp data_off) {
  viennacl::ocl::kernel &kernel = prog->get_kernel(
      CL_KERNEL_SELECT("col2im_ndsk"));

  viennacl::ocl::enqueue(
      kernel(im_size, num_spatial_axes, WrapHandle(data_col, ctx), data_col_off,
             WrapHandle(im_shape, ctx), WrapHandle(col_shape, ctx),
             WrapHandle(kernel_shape, ctx), WrapHandle(pad, ctx),
             WrapHandle(stride, ctx), WrapHandle(kstride, ctx),
             WrapHandle(data_im, ctx), data_off),
      ctx->get_queue());
}

// Explicit instantiation
template void greentea_col2im_ndsk_gpu<float>(viennacl::ocl::program *prog,
                                              viennacl::ocl::context *ctx,
                                              cl_mem data_col,
                                              const int_tp data_col_off,
                                              const int_tp num_spatial_axes,
                                              const int_tp im_size,
                                              cl_mem im_shape, cl_mem col_shape,
                                              cl_mem kernel_shape, cl_mem pad,
                                              cl_mem stride, cl_mem kstride,
                                              cl_mem data_im, int_tp data_off);

template void greentea_col2im_ndsk_gpu<double>(viennacl::ocl::program *prog,
                                               viennacl::ocl::context *ctx,
                                               cl_mem data_col,
                                               const int_tp data_col_off,
                                               const int_tp num_spatial_axes,
                                               const int_tp im_size,
                                               cl_mem im_shape,
                                               cl_mem col_shape,
                                               cl_mem kernel_shape, cl_mem pad,
                                               cl_mem stride, cl_mem kstride,
                                               cl_mem data_im, int_tp data_off);

}  // namespace caffe
#endif
