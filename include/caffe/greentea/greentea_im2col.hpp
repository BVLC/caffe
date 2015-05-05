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
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/vector.hpp"

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
                            cl_mem data_col);

/*template <typename Dtype>
 void im2col_gpu(const Dtype* data_im, const int channels,
 const int height, const int width, const int kernel_h, const int kernel_w,
 const int pad_h, const int pad_w, const int stride_h,
 const int stride_w, Dtype* data_col);

 template <typename Dtype>
 void col2im_sk_gpu(const Dtype* data_col, const int channels,
 const int height, const int width, const int patch_h, const int patch_w,
 const int pad_h, const int pad_w, const int stride_h,
 const int stride_w, const int kstride_h, const int kstride_w,
 Dtype* data_im);

 template <typename Dtype>
 void col2im_gpu(const Dtype* data_col, const int channels,
 const int height, const int width, const int patch_h, const int patch_w,
 const int pad_h, const int pad_w, const int stride_h,
 const int stride_w, Dtype* data_im);*/

}

#endif
#endif /* GREENTEA_IM2COL_HPP_ */
