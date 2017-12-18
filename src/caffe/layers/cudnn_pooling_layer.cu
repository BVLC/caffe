#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  Forward_const_gpu(bottom, top);
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Forward_const_gpu(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) const {

  if (!bottom_desc_ptr_.get()) {
    bottom_desc_ptr_.reset(new cudnnTensorDescriptor_t{});
    cudnn::createTensor4dDesc<Dtype>(bottom_desc_ptr_.get());
  }

  if (!top_desc_ptr_.get()) {
    top_desc_ptr_.reset(new cudnnTensorDescriptor_t{});
    cudnn::createTensor4dDesc<Dtype>(top_desc_ptr_.get());
  }

  if (!pooling_desc_ptr_.get()) {
    cudnnPoolingMode_t mode_;
    pooling_desc_ptr_.reset(new cudnnPoolingDescriptor_t{});
    cudnn::createPoolingDesc<Dtype>(
        pooling_desc_ptr_.get(), this->layer_param_.pooling_param().pool(),
        &mode_, this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_,
        this->stride_h_, this->stride_w_);
  }

  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();

  int pooled_height = top[0]->height();
  int pooled_width = top[0]->width();
  cudnn::setTensor4dDesc<Dtype>(bottom_desc_ptr_.get(), bottom[0]->num(),
                                bottom[0]->channels(), bottom[0]->height(),
                                bottom[0]->width());
  cudnn::setTensor4dDesc<Dtype>(top_desc_ptr_.get(), bottom[0]->num(),
                                bottom[0]->channels(), pooled_height,
                                pooled_width);

  CUDNN_CHECK(cudnnPoolingForward(
      Caffe::cudnn_handle(), *pooling_desc_ptr_, cudnn::dataType<Dtype>::one,
      *bottom_desc_ptr_, bottom_data, cudnn::dataType<Dtype>::zero,
      *top_desc_ptr_, top_data));
}

INSTANTIATE_LAYER_GPU_FUNCS_CONST(CuDNNPoolingLayer);

} // namespace caffe
#endif
