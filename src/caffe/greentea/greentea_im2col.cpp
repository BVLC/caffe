/*
 * greentea_im2col.cpp
 *
 *  Created on: Apr 8, 2015
 *      Author: Fabian Tschopp
 */

#include "caffe/greentea/greentea_im2col.hpp"

namespace caffe {

template<typename Dtype>
void greentea_im2col_sk_gpu(viennacl::ocl::program &prog,
                            viennacl::ocl::context &ctx, const cl_mem data_im,
                            const int channels, const int height,
                            const int width, const int kernel_h,
                            const int kernel_w, const int pad_h,
                            const int pad_w, const int stride_h,
                            const int stride_w, const int kstride_h,
                            const int kstride_w, cl_mem data_col) {

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
      kernel(num_kernels, WrapHandle(data_im, ctx), height, width, kernel_h,
             kernel_w, ext_kernel_h, ext_kernel_w, pad_h, pad_w, stride_h,
             stride_w, kstride_h, kstride_w, height_col, width_col,
             WrapHandle(data_col, ctx)),
      ctx.get_queue());

  std::cout << "END OF IM2COL" << std::endl;
}

// Explicit instantiation
template void greentea_im2col_sk_gpu<float>(viennacl::ocl::program &prog,
                                            viennacl::ocl::context &ctx,
                                            const cl_mem data_im,
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

}
