// Copyright 2014 George Papandreou

#ifndef CAFFE_UTIL_INTERP_H_
#define CAFFE_UTIL_INTERP_H_

#include <cublas_v2.h>
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Bi-linear interpolation
// IN : [channels height1 width1] cropped from a bigger [Height1 Width1] image
// OUT: [channels height2 width2] cropped from a bigger [Height2 Width2] image

template <typename Dtype, bool packed>
void caffe_cpu_interp2(const int channels,
    const Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
          Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2);

template <typename Dtype, bool packed>
void caffe_gpu_interp2(const int channels,
    const Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
          Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2);

// Backward (adjoint) operation
template <typename Dtype, bool packed>
void caffe_cpu_interp2_backward(const int channels,
	  Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    const Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2);

template <typename Dtype, bool packed>
void caffe_gpu_interp2_backward(const int channels,
	  Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    const Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2);

// Create Gaussian pyramid of an image. Assume output space is pre-allocated.
// IN : [channels height width]
template <typename Dtype, bool packed>
void caffe_cpu_pyramid2(const int channels,
    const Dtype *data, const int height, const int width,
    Dtype *data_pyr, const int levels);

template <typename Dtype, bool packed>
void caffe_gpu_pyramid2(const int channels,
    const Dtype *data, const int height, const int width,
    Dtype *data_pyr, const int levels);

  /*
template <typename Dtype, bool packed>
void caffe_cpu_mosaic(const int channels,
    const Dtype *data1, const MosaicParameter mosaic_params1,
    const Dtype *data_pyr, const int levels,
          Dtype *data2, const MosaicParameter mosaic_params2);

template <typename Dtype, bool packed>
void caffe_gpu_mosaic(const int channels,
    const Dtype *data1, const MosaicParameter mosaic_params1,
    const Dtype *data_pyr, const int levels,
          Dtype *data2, const MosaicParameter mosaic_params2);
  */

}  // namespace caffe

#endif
