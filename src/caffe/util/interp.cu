// Copyright 2014 George Papandreou

#include "caffe/common.hpp"
#include "caffe/common.cuh"
#include "caffe/util/interp.hpp"

namespace caffe {

// Bi-linear interpolation
// IN : [channels height1 width1] cropped from a bigger [Height1 Width1] image
// OUT: [channels height2 width2] cropped from a bigger [Height2 Width2] image
template <typename Dtype, bool packed>
__global__ void caffe_gpu_interp2_kernel(const int n, const float rheight, const float rwidth,
    const int channels,
    const Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    const int w2 = index % width2; // 0:width2-1
    const int h2 = index / width2; // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;
      if (packed) {
	const Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
	Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
	for (int c = 0; c < channels; ++c) {
	  pos2[0] = pos1[0];
	  pos1++;
	  pos2++;
	}
      }
      else {
	const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
	Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
	for (int c = 0; c < channels; ++c) {
	pos2[0] = pos1[0];
	pos1 += Width1 * Height1;
	pos2 += Width2 * Height2;
	}
      }
      return;
    }
    //
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Dtype h1lambda = h1r - h1;
    const Dtype h0lambda = Dtype(1.) - h1lambda;
    //
    const float w1r = rwidth * w2;
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Dtype w1lambda = w1r - w1;
    const Dtype w0lambda = Dtype(1.) - w1lambda;
    //
    if (packed) {
      const Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
      Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
      for (int c = 0; c < channels; ++c) {
	pos2[0] =
	  h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[channels * w1p]) + 
	  h1lambda * (w0lambda * pos1[channels * h1p * Width1] + w1lambda * pos1[channels * (h1p * Width1 + w1p)]);
	pos1++;
	pos2++;
      }
    }
    else {
      const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
      Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
      for (int c = 0; c < channels; ++c) {
	pos2[0] =
	  h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[w1p]) + 
	  h1lambda * (w0lambda * pos1[h1p * Width1] + w1lambda * pos1[h1p * Width1 + w1p]);
	pos1 += Width1 * Height1;
	pos2 += Width2 * Height2;
      }
    }
  }
}

template <typename Dtype, bool packed>
void caffe_gpu_interp2(const int channels,
    const Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
  CHECK(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
  CHECK(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2);
  const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  const int num_kernels = height2 * width2;
  caffe_gpu_interp2_kernel<Dtype,packed><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
    (num_kernels, rheight, rwidth, channels,
     data1, x1, y1, height1, width1, Height1, Width1,
     data2, x2, y2, height2, width2, Height2, Width2);
  CUDA_POST_KERNEL_CHECK;
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename Dtype, bool packed>
__global__ void caffe_gpu_interp2_kernel_backward(const int n, const float rheight, const float rwidth,
    const int channels,
    Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    const Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    const int w2 = index % width2; // 0:width2-1
    const int h2 = index / width2; // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;
      if (packed) {
	Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
	const Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
	for (int c = 0; c < channels; ++c) {
	  pos1[0] += pos2[0];
	  pos1++;
	  pos2++;
	}
      }
      else {
	Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
	const Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
	for (int c = 0; c < channels; ++c) {
	  pos1[0] += pos2[0];
	  pos1 += Width1 * Height1;
	  pos2 += Width2 * Height2;
	}
      }
      return;
    }
    //
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Dtype h1lambda = h1r - h1;
    const Dtype h0lambda = Dtype(1.) - h1lambda;
    //
    const float w1r = rwidth * w2;
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Dtype w1lambda = w1r - w1;
    const Dtype w0lambda = Dtype(1.) - w1lambda;
    //
    if (packed) {
      Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
      const Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
      for (int c = 0; c < channels; ++c) {
	atomicAdd(&pos1[0], h0lambda * w0lambda * pos2[0]);
	atomicAdd(&pos1[channels * w1p], h0lambda * w1lambda * pos2[0]);
	atomicAdd(&pos1[channels * h1p * Width1], h1lambda * w0lambda * pos2[0]);
	atomicAdd(&pos1[channels * (h1p * Width1 + w1p)], h1lambda * w1lambda * pos2[0]);
	pos1++;
	pos2++;
      }
    }
    else {
      Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
      const Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
      for (int c = 0; c < channels; ++c) {
	atomicAdd(&pos1[0], h0lambda * w0lambda * pos2[0]);
	atomicAdd(&pos1[w1p], h0lambda * w1lambda * pos2[0]);
	atomicAdd(&pos1[h1p * Width1], h1lambda * w0lambda * pos2[0]);
	atomicAdd(&pos1[h1p * Width1 + w1p], h1lambda * w1lambda * pos2[0]);
	pos1 += Width1 * Height1;
	pos2 += Width2 * Height2;
      }
    }
  }
}

template <typename Dtype, bool packed>
void caffe_gpu_interp2_backward(const int channels,
    Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    const Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
  CHECK(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
  CHECK(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2);
  const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  const int num_kernels = height2 * width2;
  caffe_gpu_interp2_kernel_backward<Dtype,packed><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
    (num_kernels, rheight, rwidth, channels,
     data1, x1, y1, height1, width1, Height1, Width1,
     data2, x2, y2, height2, width2, Height2, Width2);
  CUDA_POST_KERNEL_CHECK;
}


// Create Gaussian pyramid of an image. Assume output space is pre-allocated.
// IN : [channels height width]
template <typename Dtype, bool packed>
__global__ void caffe_gpu_pyramid2_kernel(const int n, const int channels,
    const Dtype *data1, const int height1, const int width1,
    Dtype *data2, const int height2, const int width2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    const int w2 = index % width2; // 0:width2-1
    const int h2 = index / width2; // 0:height2-1
    const int w1 = 2 * w2;
    const int h1 = 2 * h2;
    if (packed) {
      const Dtype* pos1 = &data1[channels * (h1 * width1 + w1)];
      Dtype* pos2 = &data2[channels * (h2 * width2 + w2)];
      for (int c = 0; c < channels; ++c) {
	pos2[0] =  static_cast<Dtype>(.25) *
	  (pos1[0]                 + pos1[channels] + 
	   pos1[channels * width1] + pos1[channels * (width1 + 1)]);
	pos1++;
	pos2++;
      }
    }
    else {
      const Dtype* pos1 = &data1[h1 * width1 + w1];
      Dtype* pos2 = &data2[h2 * width2 + w2];
      for (int c = 0; c < channels; ++c) {
	pos2[0] =  static_cast<Dtype>(.25) *
	  (pos1[0]      + pos1[1] + 
	   pos1[width1] + pos1[width1 + 1]);
	pos1 += width1 * height1;
	pos2 += width2 * height2;
      }
    }
  }
}

template <typename Dtype, bool packed>
void caffe_gpu_pyramid2(const int channels,
    const Dtype *data, const int height, const int width,
    Dtype *data_pyr, const int levels) {
  CHECK(height > 0 && width > 0 && levels >= 0);
  int height1 = height, width1 = width;
  int height2 = height, width2 = width;
  const Dtype *data1 = data;
  Dtype *data2 = data_pyr;
  for (int l = 0; l < levels; ++l) {
    height2 /= 2;
    width2 /= 2;
    if (height2 == 0 || width2 == 0) {
      break;
    }
    const int num_kernels = height2 * width2;
    caffe_gpu_pyramid2_kernel<Dtype,packed><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
      (num_kernels, channels, data1, height1, width1, data2, height2, width2);
    CUDA_POST_KERNEL_CHECK;
    data1 = data2;
    height1 = height2;
    width1 = width2;
    data2 += channels * height2 * width2;
  }
}


// Explicit instances
template void caffe_gpu_interp2<float,false>(const int, const float *, const int, const int, const int, const int, const int, const int, float *, const int, const int, const int, const int, const int, const int);
template void caffe_gpu_interp2<float,true>(const int, const float *, const int, const int, const int, const int, const int, const int, float *, const int, const int, const int, const int, const int, const int);
template void caffe_gpu_interp2<double,false>(const int, const double *, const int, const int, const int, const int, const int, const int, double *, const int, const int, const int, const int, const int, const int);
template void caffe_gpu_interp2<double,true>(const int, const double *, const int, const int, const int, const int, const int, const int, double *, const int, const int, const int, const int, const int, const int);

template void caffe_gpu_interp2_backward<float,false>(const int, float *, const int, const int, const int, const int, const int, const int, const float *, const int, const int, const int, const int, const int, const int);
template void caffe_gpu_interp2_backward<double,false>(const int, double *, const int, const int, const int, const int, const int, const int, const double *, const int, const int, const int, const int, const int, const int);

template void caffe_gpu_pyramid2<float,false>(const int, const float *, const int, const int, float *, const int);
template void caffe_gpu_pyramid2<float,true>(const int, const float *, const int, const int, float *, const int);
template void caffe_gpu_pyramid2<double,false>(const int, const double *, const int, const int, double *, const int);
template void caffe_gpu_pyramid2<double,true>(const int, const double *, const int, const int, double *, const int);

}  // namespace caffe
