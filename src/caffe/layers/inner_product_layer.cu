#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (M_ == 1) {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype) 1., weight,
                            bottom_data, (Dtype) 0., top_data);
      if (bias_term_)
        caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                              this->blobs_[1]->gpu_data(), top_data);
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype) 1.,
                            bottom_data, weight, (Dtype) 0., top_data);
      if (bias_term_)
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype) 1.,
                              bias_multiplier_.gpu_data(),
                              this->blobs_[1]->gpu_data(), (Dtype) 1.,
                              top_data);
    }
#endif  // USE CUDA
  } else {
#ifdef USE_GREENTEA
    if (M_ == 1) {
      greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans, N_,
                               K_, (Dtype) 1., (cl_mem) weight, 0,
                               (cl_mem) bottom_data, 0, (Dtype) 0.,
                               (cl_mem) top_data, 0);
      if (bias_term_)
        greentea_gpu_axpy<Dtype>(this->device_->id(), N_,
                                 bias_multiplier_.cpu_data()[0],
                                 (cl_mem) (this->blobs_[1]->gpu_data()), 0,
                                 (cl_mem) top_data, 0);
    } else {
      greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                               CblasTrans, M_, N_, K_, (Dtype) 1.,
                               (cl_mem) bottom_data, 0, (cl_mem) weight, 0,
                               (Dtype) 0., (cl_mem) top_data, 0);
      if (bias_term_)
        greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                                 CblasNoTrans, M_, N_, 1, (Dtype) 1.,
                                 (cl_mem) (bias_multiplier_.gpu_data()), 0,
                                 (cl_mem) (this->blobs_[1]->gpu_data()), 0,
                                 (Dtype) 1., (cl_mem) top_data, 0);
    }
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (this->param_propagate_down_[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      // Gradient with respect to weight
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype) 1.,
                            top_diff, bottom_data, (Dtype) 1.,
                            this->blobs_[0]->mutable_gpu_diff());
    }
    if (bias_term_ && this->param_propagate_down_[1]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      // Gradient with respect to bias
      caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype) 1., top_diff,
                            bias_multiplier_.gpu_data(), (Dtype) 1.,
                            this->blobs_[1]->mutable_gpu_diff());
    }
    if (propagate_down[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      // Gradient with respect to bottom data
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype) 1.,
                            top_diff, this->blobs_[0]->gpu_data(), (Dtype) 0.,
                            bottom[0]->mutable_gpu_diff());
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    if (this->param_propagate_down_[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      // Gradient with respect to weight
      greentea_gpu_gemm<Dtype>(this->device_->id(), CblasTrans,
                               CblasNoTrans, N_, K_, M_, (Dtype) 1.,
                               (cl_mem) top_diff, 0, (cl_mem) bottom_data, 0,
                               (Dtype) 1.,
                               (cl_mem) (this->blobs_[0]->mutable_gpu_diff()),
                               0);
    }
    if (bias_term_ && this->param_propagate_down_[1]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      // Gradient with respect to bias
      greentea_gpu_gemv<Dtype>(this->device_->id(), CblasTrans, M_, N_,
                               (Dtype) 1., (cl_mem) top_diff, 0,
                               (cl_mem) (bias_multiplier_.gpu_data()), 0,
                               (Dtype) 1.,
                               (cl_mem) (this->blobs_[1]->mutable_gpu_diff()),
                               0);
    }
    if (propagate_down[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      // Gradient with respect to bottom data
      greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                               CblasNoTrans, M_, K_, N_, (Dtype) 1.,
                               (cl_mem) top_diff, 0,
                               (cl_mem) (this->blobs_[0]->gpu_data()), 0,
                               (Dtype) 0.,
                               (cl_mem) (bottom[0]->mutable_gpu_diff()), 0);
    }
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
