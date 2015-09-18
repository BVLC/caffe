/*
 * greentea_im2col.hpp
 *
 *  Created on: Apr 8, 2015
 *      Author: fabian
 */

#ifndef GREENTEA_IM2COL_HPP_
#define GREENTEA_IM2COL_HPP_
#ifdef USE_GREENTEA

#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/vector.hpp"

namespace caffe {

template<typename Dtype>
void greentea_im2col_sk_gpu(viennacl::ocl::program *prog,
                            viennacl::ocl::context *ctx,
                            const cl_mem data_im, const int data_offset,
                            const int channels, const int height,
                            const int width, const int kernel_h,
                            const int kernel_w, const int pad_h,
                            const int pad_w, const int stride_h,
                            const int stride_w, const int kstride_h,
                            const int kstride_w, cl_mem data_col);

template<typename Dtype>
void greentea_col2im_sk_gpu(viennacl::ocl::program *prog,
                            viennacl::ocl::context *ctx,
                            const cl_mem data_col, const int channels,
                            const int height, const int width,
                            const int patch_h, const int patch_w,
                            const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int kstride_h, const int kstride_w,
                            cl_mem data_im, const int data_offset);

template<typename Dtype>
void greentea_im2col_gpu(viennacl::ocl::program *prog,
                         viennacl::ocl::context *ctx,
                         const cl_mem data_im, const int data_im_off,
                         const int channels, const int height, const int width,
                         const int kernel_h, const int kernel_w,
                         const int pad_h, const int pad_w, const int stride_h,
                         const int stride_w, cl_mem data_col,
                         const int data_col_off);

template<typename Dtype>
void greentea_col2im_gpu(viennacl::ocl::program *prog,
                         viennacl::ocl::context *ctx,
                         const cl_mem data_col, const int data_col_off,
                         const int channels, const int height, const int width,
                         const int patch_h, const int patch_w, const int pad_h,
                         const int pad_w, const int stride_h,
                         const int stride_w, cl_mem data_im,
                         const int data_im_off);

template<typename Dtype>
void greentea_im2col_nd_gpu(viennacl::ocl::program *prog,
                            viennacl::ocl::context *ctx, cl_mem data_im,
                            const int data_off, const int num_spatial_axes,
                            const int num_kernels,
                            cl_mem im_shape, cl_mem col_shape,
                            cl_mem kernel_shape, cl_mem pad, cl_mem stride,
                            cl_mem kstride, cl_mem data_col, int data_col_off);

template<typename Dtype>
void greentea_col2im_nd_gpu(viennacl::ocl::program *prog,
                            viennacl::ocl::context *ctx, cl_mem data_col,
                            const int data_col_off, const int num_spatial_axes,
                            const int im_size, cl_mem im_shape,
                            cl_mem col_shape, cl_mem kernel_shape, cl_mem pad,
                            cl_mem stride, cl_mem kstride, cl_mem data_im,
                            int data_off);



}  // namespace caffe

#endif  // USE_GREENTEA
#endif  /* GREENTEA_IM2COL_HPP_ */
