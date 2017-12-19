#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/filler.hpp"
#include "caffe/layers/normalize2_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// divid a matrix with vector
template <typename Dtype>
__global__ void DivBsx(const int nthreads, const Dtype* A,
    const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
    Dtype* B) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = index % cols;
    int r = (index / cols) % rows;
    if (trans == CblasNoTrans) {
      B[index] = A[index] / v[c];
    } else {
      B[index] = A[index] / v[r];
    }
  }
}

template <typename Dtype>
__global__ void MulBsx(const int nthreads, const Dtype* A,
    const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
    Dtype* B) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = index % cols;
    int r = (index / cols) % rows;
    if (trans == CblasNoTrans) {
      B[index] = A[index] * v[c];
    } else {
      B[index] = A[index] * v[r];
    }
  }
}

template <typename Dtype>
void Normalize2Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_const_gpu(bottom,top);
}

template <typename Dtype>
void Normalize2Layer<Dtype>::Forward_const_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) const {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  int channels = bottom[0]->channels();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Blob<Dtype> norm;
  Dtype* norm_data;
  if (across_spatial_) {
    // need to index it
    norm.Reshape(bottom[0]->num(), 1, 1, 1);
    norm_data = norm.mutable_cpu_data();
  } else {
    norm.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
    norm_data = norm.mutable_gpu_data();
    // add eps to avoid overflow
    caffe_gpu_set<Dtype>(norm.count(), Dtype(eps_), norm_data);
  }
  const Dtype* scale;
  if (channel_shared_) {
    scale = this->blobs_[0]->cpu_data();
  } else {
    scale = this->blobs_[0]->gpu_data();
  }
  Blob<Dtype> sum_channel_multiplier;
  sum_channel_multiplier.Reshape(1, channels, 1, 1);
  caffe_set(channels, Dtype(1), sum_channel_multiplier.mutable_cpu_data());
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  Blob<Dtype> buffer;
  buffer.Reshape(1, 1,1,dim);
  Dtype* buffer_data = buffer.mutable_gpu_data();
  for (int n = 0; n < num; ++n) {
    caffe_gpu_powx<Dtype>(dim, bottom_data, Dtype(2), buffer_data);
    if (across_spatial_) {
      Dtype normsqr;
      caffe_gpu_asum<Dtype>(dim, buffer_data, &normsqr);
      // add eps to avoid overflow
      norm_data[n] = pow(normsqr+eps_, Dtype(0.5));
      caffe_gpu_scale<Dtype>(dim, Dtype(1.0 / norm_data[n]), bottom_data,
                             top_data);
    } else {
      // compute norm
      caffe_gpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, Dtype(1),
                            buffer_data,sum_channel_multiplier.gpu_data(), Dtype(1),
                            norm_data);
      caffe_gpu_powx<Dtype>(spatial_dim, norm_data, Dtype(0.5), norm_data);
      // scale the layer
      // NOLINT_NEXT_LINE(whitespace/operators)
      DivBsx<Dtype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
          dim, bottom_data, norm_data, channels, spatial_dim, CblasNoTrans,
          top_data);
      CUDA_POST_KERNEL_CHECK;
      norm_data += spatial_dim;
    }
    // scale the output
    if (channel_shared_) {
      caffe_gpu_scal<Dtype>(dim, scale[0], top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      MulBsx<Dtype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
          dim, top_data, scale, channels, spatial_dim, CblasTrans,
          top_data);
      CUDA_POST_KERNEL_CHECK;
    }
    bottom_data += dim;
    top_data += dim;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS_CONST(Normalize2Layer);


}  // namespace caffe
