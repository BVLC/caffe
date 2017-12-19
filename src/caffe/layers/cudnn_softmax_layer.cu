#ifdef USE_CUDNN
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/cudnn_softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  Forward_const_gpu(bottom, top);
}

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Forward_const_gpu(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) const {
  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();

  auto softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  int N = bottom[0]->count(0, softmax_axis_);
  int K = bottom[0]->shape(softmax_axis_);
  int H = bottom[0]->count(softmax_axis_ + 1);
  int W = 1;

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

  CUDNN_CHECK(cudnnSoftmaxForward(
      Caffe::cudnn_handle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
      cudnn::dataType<Dtype>::one, *bottom_desc_ptr_, bottom_data,
      cudnn::dataType<Dtype>::zero, *top_desc_ptr_, top_data));
}

INSTANTIATE_LAYER_GPU_FUNCS_CONST(CuDNNSoftmaxLayer);

} // namespace caffe
#endif
