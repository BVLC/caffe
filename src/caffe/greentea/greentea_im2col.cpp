/*
 * greentea_im2col.cpp
 *
 *  Created on: Apr 8, 2015
 *      Author: Fabian Tschopp
 */
#ifdef USE_GREENTEA
#include "caffe/greentea/greentea_im2col.hpp"

namespace caffe {

template<typename Dtype>
void greentea_im2col_sk_gpu(viennacl::ocl::program &prog,
                            viennacl::ocl::context &ctx, const cl_mem data_im,
                            const int data_offset, const int channels,
                            const int height, const int width,
                            const int kernel_h, const int kernel_w,
                            const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int kstride_h, const int kstride_w,
                            cl_mem data_col) {

  std::cout << "DATA_IM: " << data_im << std::endl;
  std::cout << "DATA_COL: " << data_col << std::endl;

  int ext_kernel_h = (kernel_h - 1) * kstride_h + 1;
  int ext_kernel_w = (kernel_w - 1) * kstride_w + 1;
  int height_col = (height + 2 * pad_h - ext_kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - ext_kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;

  viennacl::ocl::kernel &kernel = prog.get_kernel(
      CL_KERNEL_SELECT("im2col_sk_gpu_kernel"));

  viennacl::ocl::enqueue(
      kernel(num_kernels, WrapHandle(data_im, ctx), data_offset, height, width,
             kernel_h, kernel_w, ext_kernel_h, ext_kernel_w, pad_h, pad_w,
             stride_h, stride_w, kstride_h, kstride_w, height_col, width_col,
             WrapHandle(data_col, ctx)),
      ctx.get_queue());

  std::cout << "END OF IM2COL" << std::endl;
}

// Explicit instantiation
template void greentea_im2col_sk_gpu<float>(viennacl::ocl::program &prog,
                                            viennacl::ocl::context &ctx,
                                            const cl_mem data_im,
                                            const int data_offset,
                                            const int channels,
                                            const int height, const int width,
                                            const int kernel_h,
                                            const int kernel_w, const int pad_h,
                                            const int pad_w, const int stride_h,
                                            const int stride_w,
                                            const int kstride_h,
                                            const int kstride_w,
                                            cl_mem data_col);

template void greentea_im2col_sk_gpu<double>(viennacl::ocl::program &prog,
                                             viennacl::ocl::context &ctx,
                                             const cl_mem data_im,
                                             const int data_offset,
                                             const int channels,
                                             const int height, const int width,
                                             const int kernel_h,
                                             const int kernel_w,
                                             const int pad_h, const int pad_w,
                                             const int stride_h,
                                             const int stride_w,
                                             const int kstride_h,
                                             const int kstride_w,
                                             cl_mem data_col);

template<typename Dtype>
void greentea_col2im_sk_gpu(viennacl::ocl::program &prog,
                            viennacl::ocl::context &ctx, const cl_mem data_col,
                            const int channels, const int height,
                            const int width, const int patch_h,
                            const int patch_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int kstride_h, const int kstride_w,
                            cl_mem data_im, const int data_offset) {

  if (stride_w > 1 || stride_h > 1 || pad_h > 0 || pad_w > 0) {
    LOG(FATAL)<<"stride greater than 1 or pad greater than 0 not tested in col2im_sk_gpu().";
  }

  int ext_patch_h = (patch_h - 1) * kstride_h + 1;
  int ext_patch_w = (patch_w - 1) * kstride_w + 1;
  int height_col = (height + 2 * pad_h - ext_patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - ext_patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;

  viennacl::ocl::kernel &kernel = prog.get_kernel(
      CL_KERNEL_SELECT("col2im_sk"));

  viennacl::ocl::enqueue(
      kernel(num_kernels, num_kernels, WrapHandle(data_col,ctx), height, width, channels,
          patch_h, patch_w, ext_patch_h, ext_patch_w,
          pad_h, pad_w, stride_h, stride_w, kstride_h, kstride_w,
          height_col, width_col, WrapHandle(data_im,ctx), data_offset),
      ctx.get_queue());

}

template void greentea_col2im_sk_gpu<float>(viennacl::ocl::program &prog,
                                            viennacl::ocl::context &ctx,
                                            const cl_mem data_col,
                                            const int channels,
                                            const int height, const int width,
                                            const int patch_h,
                                            const int patch_w, const int pad_h,
                                            const int pad_w, const int stride_h,
                                            const int stride_w,
                                            const int kstride_h,
                                            const int kstride_w, cl_mem data_im,
                                            const int data_offset);

template void greentea_col2im_sk_gpu<double>(viennacl::ocl::program &prog,
                                             viennacl::ocl::context &ctx,
                                             const cl_mem data_col,
                                             const int channels,
                                             const int height, const int width,
                                             const int patch_h,
                                             const int patch_w, const int pad_h,
                                             const int pad_w,
                                             const int stride_h,
                                             const int stride_w,
                                             const int kstride_h,
                                             const int kstride_w,
                                             cl_mem data_im,
                                             const int data_offset);

}
#endif
