#ifdef USE_FFT
#include <algorithm>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/fft.hpp"

namespace caffe {

// copy weights to buffer for fft in-place
template <typename Dtype>
__global__ void fft_gpu_copy2buffer_kernel(Dtype* fft_gpu_weights_real_,
         const Dtype* weight, const int ch_gr, const int ker_h_,
         const int ker_w_, const int fft_height_,
         const int fft_width_) {
  int out = blockIdx.x;
  int c = blockIdx.y * blockDim.x +threadIdx.x;
  int map_offset = out * ch_gr + c;
  if (c < ch_gr) {
    for (int h = 0; h < ker_h_; h++) {
      for (int w = 0; w < ker_w_; w++) {
        fft_gpu_weights_real_[(map_offset * fft_height_ + h) * 2 *
              (fft_width_/2+1) + w] =
              weight[(map_offset* ker_h_+ h) * ker_w_ + w];
      }
    }
  }
}

template <typename Dtype>
__global__ void fft_gpu_copy2buffer2D_in_kernel(Dtype* map_out,
         const Dtype* map_in, int width_out, int height_, int width_,
         int stride_h_, int stride_w_, int pad_h_, int pad_w_, Dtype units) {
  int h = blockIdx.x;
  int w = threadIdx.x;
  map_out[(h*stride_h_ + pad_h_)* width_out + (w*stride_w_ + pad_w_)] =
        units*map_in[h * width_ + w];
}

template <typename Dtype>
__global__ void fft_gpu_copy2buffer2D_out_kernel(Dtype* map_out,
         const Dtype* map_in, int height_out_, int width_out_, int fft_height_,
         int fft_width_, int stride_h_, int stride_w_,
         int pad_h_, int pad_w_, Dtype units) {
  int h_out = blockIdx.x;
  int w_out = threadIdx.x;
  int h = h_out * stride_h_ + pad_h_;
  int w = w_out * stride_w_ + pad_w_;
  if ((h < fft_height_) && (w < fft_width_)) {
      map_out[h_out*width_out_ + w_out] =
            (units * map_in[h*fft_width_ + w]);
  }
}

// TODO remove (i < size)
__global__ void caffe_gpu_elementMulConj_kernel(cufftComplex* dst,
         cufftComplex* src1, cufftComplex* src2, int size) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    dst[i].x +=  src1[i].x * src2[i].x + src1[i].y * src2[i].y;
    dst[i].y += -src1[i].x * src2[i].y + src1[i].y * src2[i].x;
  }
}

__global__ void caffe_gpu_elementMulConj_double_kernel(cufftDoubleComplex* dst,
         cufftDoubleComplex* src1, cufftDoubleComplex* src2, int size) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    dst[i].x +=  src1[i].x * src2[i].x + src1[i].y * src2[i].y;
    dst[i].y += -src1[i].x * src2[i].y + src1[i].y * src2[i].x;
  }
}

__global__ void caffe_gpu_elementMul_kernel(cufftComplex* dst,
         cufftComplex* src1, cufftComplex* src2, int size) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    dst[i].x += src1[i].x * src2[i].x - src1[i].y * src2[i].y;
    dst[i].y += src1[i].x * src2[i].y + src1[i].y * src2[i].x;
  }
}

__global__ void caffe_gpu_elementMul_double_kernel(cufftDoubleComplex* dst,
         cufftDoubleComplex* src1, cufftDoubleComplex* src2, int size) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    dst[i].x += src1[i].x * src2[i].x - src1[i].y * src2[i].y;
    dst[i].y += src1[i].x * src2[i].y + src1[i].y * src2[i].x;
  }
}

// copy weights to fft_weight buffer, one thread per one filter
template <typename Dtype>
void fft_gpu_copy2buffer(Dtype* fft_gpu_weights_real_,
         const Dtype* weight, int num_output_, int group_,
         int channels_, int ker_h_, int ker_w_,
          int fft_height_, int fft_width_) {
  int ch_gr = channels_/group_;
  dim3 block_num(num_output_, (ch_gr/CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;
  // int block_num = num_output_;
  // int thread_num = channels_/group_;
  fft_gpu_copy2buffer_kernel <<< block_num, thread_num>>>
           (fft_gpu_weights_real_, weight, ch_gr, ker_h_,
              ker_w_, fft_height_, fft_width_);
  CUDA_POST_KERNEL_CHECK;
}


// copy from bottom (top) to buffer, one thread per one data element
// TODO: compare with kernel, when one thread copy the whole row/column
template <typename Dtype>
void fft_gpu_copy2buffer2D_in(Dtype* map_out, const Dtype* map_in,
         int width_out, int height_, int width_,
         int stride_h_, int stride_w_, int pad_h_, int pad_w_, Dtype units) {
  fft_gpu_copy2buffer2D_in_kernel <<< height_, width_>>>(map_out, map_in,
      width_out, height_, width_, stride_h_, stride_w_, pad_h_, pad_w_, units);
  CUDA_POST_KERNEL_CHECK;
}

// copy from buffer to top(bottom), one thread per one data element
template <typename Dtype>
void fft_gpu_copy2buffer2D_out(Dtype* map_out, const Dtype* map_in,
         int height_out_, int width_out_, int fft_height_, int fft_width_,
         int stride_h_, int stride_w_, int pad_h_, int pad_w_, Dtype units) {
  fft_gpu_copy2buffer2D_out_kernel <<< height_out_, width_out_>>>(map_out,
        map_in, height_out_, width_out_, fft_height_, fft_width_, stride_h_,
        stride_w_, pad_h_, pad_w_, units);
  CUDA_POST_KERNEL_CHECK;
}

template void fft_gpu_copy2buffer2D_in(float* map_out, const float* map_in,
    int width_out, int height_, int width_,
    int stride_h_, int stride_w_, int pad_h_, int pad_w_, float units);
template void fft_gpu_copy2buffer2D_in(double* map_out, const double* map_in,
    int width_out, int height_, int width_,
    int stride_h_, int stride_w_, int pad_h_, int pad_w_, double units);

template void fft_gpu_copy2buffer2D_out(float* map_out, const float* map_in,
    int height_out_, int width_out_, int fft_height_, int fft_width_,
    int stride_h_, int stride_w_, int pad_h_, int pad_w_, float units);
template void fft_gpu_copy2buffer2D_out(double* map_out, const double* map_in,
    int height_out_, int width_out_, int fft_height_, int fft_width_,
    int stride_h_, int stride_w_, int pad_h_, int pad_w_, double units);


template <>
void caffe_gpu_elementMulConj(std::complex<float>* dst,
       std::complex<float>* src1, std::complex<float>* src2, int size) {
  int blocks = size/CAFFE_CUDA_NUM_THREADS+1;
  cufftComplex* dst_cuda  = reinterpret_cast<cufftComplex*> (dst);
  cufftComplex* src1_cuda = reinterpret_cast<cufftComplex*> (src1);
  cufftComplex* src2_cuda = reinterpret_cast<cufftComplex*> (src2);
  caffe_gpu_elementMulConj_kernel <<< blocks, CAFFE_CUDA_NUM_THREADS>>>(
    dst_cuda, src1_cuda, src2_cuda, size);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_elementMulConj(std::complex<double>* dst,
       std::complex<double>* src1, std::complex<double>* src2, int size) {
  int blocks = size/CAFFE_CUDA_NUM_THREADS+1;
  cufftDoubleComplex* dst_cuda  = reinterpret_cast<cufftDoubleComplex*> (dst);
  cufftDoubleComplex* src1_cuda = reinterpret_cast<cufftDoubleComplex*> (src1);
  cufftDoubleComplex* src2_cuda = reinterpret_cast<cufftDoubleComplex*> (src2);
  caffe_gpu_elementMulConj_double_kernel <<< blocks,
    CAFFE_CUDA_NUM_THREADS>>>(dst_cuda, src1_cuda, src2_cuda, size);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_elementMul(std::complex<float>* dst,
    std::complex<float>* src1, std::complex<float>* src2, int size) {
  int blocks = size/CAFFE_CUDA_NUM_THREADS + 1;
  cufftComplex* dst_cuda  = reinterpret_cast<cufftComplex*> (dst);
  cufftComplex* src1_cuda = reinterpret_cast<cufftComplex*> (src1);
  cufftComplex* src2_cuda = reinterpret_cast<cufftComplex*> (src2);
  caffe_gpu_elementMul_kernel <<< blocks, CAFFE_CUDA_NUM_THREADS>>>(dst_cuda,
    src1_cuda, src2_cuda, size);
  CUDA_POST_KERNEL_CHECK;
}
template <>
void caffe_gpu_elementMul(std::complex<double>* dst,
    std::complex<double>* src1, std::complex<double>* src2, int size) {
  int blocks = size/CAFFE_CUDA_NUM_THREADS + 1;
  cufftDoubleComplex* dst_cuda  = reinterpret_cast<cufftDoubleComplex*> (dst);
  cufftDoubleComplex* src1_cuda = reinterpret_cast<cufftDoubleComplex*> (src1);
  cufftDoubleComplex* src2_cuda = reinterpret_cast<cufftDoubleComplex*> (src2);
  caffe_gpu_elementMul_double_kernel <<< blocks, CAFFE_CUDA_NUM_THREADS>>>(
    dst_cuda, src1_cuda, src2_cuda, size);
  CUDA_POST_KERNEL_CHECK;
}

template void fft_gpu_copy2buffer<float>(float* fft_gpu_weights_real_,
    const float* weight, int num_output_, int group_, int channels_,
    int ker_h_, int ker_w_, int fft_height_, int fft_width_);
template void fft_gpu_copy2buffer<double>(double* fft_gpu_weights_real_,
    const double* weight, int num_output_, int group_,
    int channels_, int ker_h_, int ker_w_, int fft_height_, int fft_width_);

template <>
void caffe_gpu_fft_execute_dft_r2c(cufftHandle plan,
    float *in, std::complex<float> *out) {
  if (cufftExecR2C(plan, reinterpret_cast<cufftReal*>(in),
      reinterpret_cast<cufftComplex*>(out)) != CUFFT_SUCCESS) {
    LOG(ERROR) <<  "cufftExecR2C error: FFT of inputs failed";
  }
}

template <>
void caffe_gpu_fft_execute_dft_r2c(cufftHandle plan,
    double *in, std::complex<double> *out) {
  if (cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal*>(in),
      reinterpret_cast<cufftDoubleComplex*>(out)) != CUFFT_SUCCESS) {
    LOG(ERROR) <<  "cufftExecD2Z error: FFT of inputs failed";
  }
}

template <>
void caffe_gpu_fft_execute_dft_c2r(cufftHandle plan,
    std::complex<float> *in, float *out) {
  if (cufftExecC2R(plan, reinterpret_cast<cufftComplex*>(in),
      reinterpret_cast<cufftReal*>(out)) != CUFFT_SUCCESS) {
    LOG(ERROR) <<  "cufftExecC2R error: FFT of inputs failed";
  }
}

template <>
void caffe_gpu_fft_execute_dft_c2r(cufftHandle plan,
    std::complex<double> *in, double *out) {
  if (cufftExecZ2D(plan, reinterpret_cast<cufftDoubleComplex*>(in),
       reinterpret_cast<cufftDoubleReal*>(out)) != CUFFT_SUCCESS) {
    LOG(ERROR) <<  "cufftExecZ2D error: FFT of inputs failed";
  }
}
template <>
void caffe_gpu_fft_execute_dft_r2c_inplace(cufftHandle plan,
    std::complex<float> *inout) {
  if (cufftExecR2C(plan, reinterpret_cast<cufftReal*>(inout),
              reinterpret_cast<cufftComplex*>(inout)) != CUFFT_SUCCESS) {
    LOG(ERROR) <<  "cufftExecR2C error: FFT of weights failed";
  }
}

template <>
void caffe_gpu_fft_execute_dft_r2c_inplace(cufftHandle plan,
    std::complex<double> *inout) {
  if (cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal*>(inout),
       reinterpret_cast<cufftDoubleComplex*>(inout)) != CUFFT_SUCCESS) {
    LOG(ERROR) <<  "cufftExecD2Z error: FFT of weights failed";
  }
}


}  // namespace caffe
#endif
