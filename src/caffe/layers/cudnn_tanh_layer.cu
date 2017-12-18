#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_tanh_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNTanHLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  Forward_const_gpu(bottom, top);
}

template <typename Dtype>
void CuDNNTanHLayer<Dtype>::Forward_const_gpu(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) const {
  const int N = bottom[0]->num();
  const int K = bottom[0]->channels();
  const int H = bottom[0]->height();
  const int W = bottom[0]->width();
  if (!bottom_desc_ptr_.get()) {
    bottom_desc_ptr_.reset(new cudnnTensorDescriptor_t{});
    cudnn::createTensor4dDesc<Dtype>(bottom_desc_ptr_.get());
  }
  cudnn::setTensor4dDesc<Dtype>(bottom_desc_ptr_.get(), N, K, H, W);

  if (!top_desc_ptr_.get()) {
    top_desc_ptr_.reset(new cudnnTensorDescriptor_t{});
    cudnn::createTensor4dDesc<Dtype>(top_desc_ptr_.get());
  }
  cudnn::setTensor4dDesc<Dtype>(top_desc_ptr_.get(), N, K, H, W);

  if (!activ_desc_ptr_.get()) {
    activ_desc_ptr_.reset(new cudnnActivationDescriptor_t{});
    cudnn::createActivationDescriptor<Dtype>(activ_desc_ptr_.get(),
                                             CUDNN_ACTIVATION_TANH);
  }

  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnActivationForward(
      Caffe::cudnn_handle(), *activ_desc_ptr_, cudnn::dataType<Dtype>::one,
      *bottom_desc_ptr_, bottom_data, cudnn::dataType<Dtype>::zero,
      *top_desc_ptr_, top_data));
#else
  CUDNN_CHECK(cudnnActivationForward_v4(
      Caffe::cudnn_handle(), *activ_desc_ptr_, cudnn::dataType<Dtype>::one,
      *bottom_desc_ptr_, bottom_data, cudnn::dataType<Dtype>::zero,
      *top_desc_ptr_, top_data));
#endif
}

INSTANTIATE_LAYER_GPU_FUNCS_CONST(CuDNNTanHLayer);

} // namespace caffe
#endif
