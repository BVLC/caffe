#include <algorithm>
#include <cfloat>
#include <vector>

#ifdef USE_CUDA
#include "thrust/device_vector.h"
#endif
#include "caffe/filler.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
#ifdef USE_CUDA
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
#endif
template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* buffer_data = buffer_.mutable_gpu_data();
  Dtype* norm_data;
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (across_spatial_) {
      // need to index it
      norm_data = norm_.mutable_cpu_data();
    } else {
      norm_data = norm_.mutable_gpu_data();
      // add eps to avoid overflow
      caffe_gpu_set<Dtype>(norm_.count(), Dtype(eps_), norm_data);
    }
    const Dtype* scale;
    if (channel_shared_) {
      scale = this->blobs_[0]->cpu_data();
    } else {
      scale = this->blobs_[0]->gpu_data();
    }
    const Dtype* sum_channel_multiplier = sum_channel_multiplier_.gpu_data();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / num;
    int spatial_dim = bottom[0]->height() * bottom[0]->width();
    int channels = bottom[0]->channels();
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
                              buffer_data, sum_channel_multiplier, Dtype(1),
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
#endif //USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());

    viennacl::ocl::program &program = this->device_->template program<Dtype>();
    if (across_spatial_) {
      // need to index it
      norm_data = norm_.mutable_cpu_data();
    } else {
      norm_data = norm_.mutable_gpu_data();
      // add eps to avoid overflow
      greentea_gpu_set<Dtype>(this->device_->id(), norm_.count(),
                              static_cast<Dtype>(eps_), (cl_mem)norm_data, 0);
    }
    const Dtype* scale;
    if (channel_shared_) {
      scale = this->blobs_[0]->cpu_data();
    } else {
      scale = this->blobs_[0]->gpu_data();
    }
    const Dtype* sum_channel_multiplier = sum_channel_multiplier_.gpu_data();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / num;
    int spatial_dim = bottom[0]->height() * bottom[0]->width();
    int channels = bottom[0]->channels();
    for (int n = 0; n < num; ++n) {
      greentea_gpu_powx<Dtype>(this->device_->id(), dim, (cl_mem)bottom_data,
                               n*dim, Dtype(2), (cl_mem)buffer_data, 0);
      if (across_spatial_) {
        Dtype normsqr;
        greentea_gpu_asum<Dtype>(this->device_->id(), dim, (cl_mem)buffer_data,
                                 0, &normsqr);
        // add eps to avoid overflow
        norm_data[n] = pow(normsqr+eps_, Dtype(0.5));
        greentea_gpu_scale<Dtype>(this->device_->id(), dim, Dtype(1.0 / norm_data[n]),
                                  (cl_mem)bottom_data, n*dim,
                                  (cl_mem)top_data, n*dim);
      } else {
        // compute norm
        greentea_gpu_gemv<Dtype>(this->device_->id(), CblasTrans, channels, spatial_dim,
                                 1., (cl_mem)buffer_data, 0, (cl_mem)sum_channel_multiplier,
                                 0, 1., (cl_mem)norm_data, n*spatial_dim);
        greentea_gpu_powx<Dtype>(this->device_->id(), spatial_dim, (cl_mem)norm_data,
                                 n*spatial_dim, 0.5, (cl_mem)norm_data, n*spatial_dim);
        // scale the layer
        //TODO
        // NOLINT_NEXT_LINE(whitespace/operators)
        // DivBsx<Dtype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
        //     dim, bottom_data, norm_data, channels, spatial_dim, CblasNoTrans,
        //    top_data);
        viennacl::ocl::kernel &oclk_divbsx = program.get_kernel(
            CL_KERNEL_SELECT("DivBsx"));
        viennacl::ocl::enqueue(
            oclk_divbsx(dim,
            WrapHandle((cl_mem) bottom_data, &ctx),
            n*dim,
            WrapHandle((cl_mem)norm_data, &ctx),
            n*spatial_dim,
            channels,
            spatial_dim,
            WrapHandle((cl_mem) top_data, &ctx),
            n*dim),
            ctx.get_queue());
      }
      // scale the output
      if (channel_shared_) {
        greentea_gpu_scal<Dtype>(this->device_->id(), dim, scale[0], (cl_mem)top_data, n*dim);
      } else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        // MulBsx<Dtype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
        //     dim, top_data, scale, channels, spatial_dim, CblasTrans,
        //     top_data);
        viennacl::ocl::kernel &oclk_mulbsx = program.get_kernel(
            CL_KERNEL_SELECT("MulBsx"));
        viennacl::ocl::enqueue(
            oclk_mulbsx(dim,
            WrapHandle((cl_mem) top_data, &ctx),
            n*dim,
            WrapHandle((cl_mem) scale, &ctx),
            channels,
            spatial_dim,
            1,
            WrapHandle((cl_mem) top_data, &ctx),
            n*dim),
            ctx.get_queue());
      }
    }
#endif //USE_GREENTEA
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
#ifdef USE_CUDA
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->mutable_gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* norm_data;
  if (across_spatial_) {
    // need to index it
    norm_data = norm_.cpu_data();
  } else {
    norm_data = norm_.gpu_data();
  }
  const Dtype* scale;
  if (channel_shared_) {
    scale = this->blobs_[0]->cpu_data();
  } else {
    scale = this->blobs_[0]->gpu_data();
  }
  Dtype* buffer_data = buffer_.mutable_gpu_data();
  Dtype* buffer_channel = buffer_channel_.mutable_gpu_data();
  Dtype* buffer_spatial = buffer_spatial_.mutable_gpu_data();
  const Dtype* sum_channel_multiplier = sum_channel_multiplier_.gpu_data();
  const Dtype* sum_spatial_multiplier = sum_spatial_multiplier_.gpu_data();
  int count = top[0]->count();
  int num = top[0]->num();
  int dim = count / num;
  int spatial_dim = top[0]->height() * top[0]->width();
  int channels = top[0]->channels();

  // Propagate to param
  if (this->param_propagate_down_[0]) {
    if (channel_shared_) {
      Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
      Dtype a;
      caffe_gpu_dot<Dtype>(count, top_data, top_diff, &a);
      scale_diff[0] += a / scale[0];
    } else {
      Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
      for (int n = 0; n < num; ++n) {
        // compute a
        caffe_gpu_mul<Dtype>(dim, top_data+n*dim, top_diff+n*dim, buffer_data);
        caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, spatial_dim, Dtype(1),
                              buffer_data, sum_spatial_multiplier, Dtype(0),
                              buffer_channel);
        // store a / scale[i] in buffer_data temporary
        caffe_gpu_div<Dtype>(channels, buffer_channel, scale, buffer_channel);
        caffe_gpu_add<Dtype>(channels, buffer_channel, scale_diff, scale_diff);
      }
    }
  }

  // Propagate to bottom
  if (propagate_down[0]) {
    for (int n = 0; n < num; ++n) {
      if (across_spatial_) {
        Dtype a;
        caffe_gpu_dot<Dtype>(dim, bottom_data, top_diff, &a);
        caffe_gpu_scale<Dtype>(dim, a / norm_data[n] / norm_data[n],
                               bottom_data, bottom_diff);
        caffe_gpu_sub<Dtype>(dim, top_diff, bottom_diff, bottom_diff);
        caffe_gpu_scale<Dtype>(dim, Dtype(1.0 / norm_data[n]), bottom_diff,
                               bottom_diff);
      } else {
        // dot product between bottom_data and top_diff
        caffe_gpu_mul<Dtype>(dim, bottom_data, top_diff, buffer_data);
        caffe_gpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, Dtype(1),
                              buffer_data, sum_channel_multiplier, Dtype(0),
                              buffer_spatial);
        // scale botom_diff
        // NOLINT_NEXT_LINE(whitespace/operators)
        MulBsx<Dtype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
            dim, bottom_data, buffer_spatial, channels, spatial_dim,
            CblasNoTrans, bottom_diff);
        CUDA_POST_KERNEL_CHECK;
        // divide by square of norm
        caffe_gpu_powx<Dtype>(spatial_dim, norm_data, Dtype(2), buffer_spatial);
        // NOLINT_NEXT_LINE(whitespace/operators)
        DivBsx<Dtype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
            dim, bottom_diff, buffer_spatial, channels, spatial_dim,
            CblasNoTrans, bottom_diff);
        CUDA_POST_KERNEL_CHECK;
        caffe_gpu_sub<Dtype>(dim, top_diff, bottom_diff, bottom_diff);
        // divide by norm
        // NOLINT_NEXT_LINE(whitespace/operators)
        DivBsx<Dtype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
            dim, bottom_diff, norm_data, channels, spatial_dim, CblasNoTrans,
            bottom_diff);
        CUDA_POST_KERNEL_CHECK;
        norm_data += spatial_dim;
      }
      // scale the diff
      if (channel_shared_) {
        caffe_gpu_scal<Dtype>(dim, scale[0], bottom_diff);
      } else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        MulBsx<Dtype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
            dim, bottom_diff, scale, channels, spatial_dim, CblasTrans,
            bottom_diff);
        CUDA_POST_KERNEL_CHECK;
      }
      bottom_data += dim;
      top_diff += dim;
      bottom_diff += dim;
    }
  }
#else
  this->Backward_cpu(top, propagate_down, bottom);
#endif //USE_CUDA
}

INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLayer);


}  // namespace caffe
