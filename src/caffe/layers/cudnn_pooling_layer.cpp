#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensorNdDesc<Dtype>(&bottom_desc_);
  cudnn::createTensorNdDesc<Dtype>(&top_desc_);
  PoolingLayer<Dtype>::LayerSetUp(bottom, top);

  const int_tp* kernel_data = this->kernel_shape_.cpu_data();
  const int_tp* pad_data = this->pad_.cpu_data();
  const int_tp* stride_data = this->stride_.cpu_data();

  cudnn::createPoolingDesc<Dtype>(&pooling_desc_,
      this->layer_param_.pooling_param().pool(), &mode_,
      this->num_spatial_axes_,
      kernel_data, pad_data, stride_data);
  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::Reshape(bottom, top);

  cudnn::setTensorNdDesc<Dtype>(&bottom_desc_,
                                bottom[0]->shape().size() - 2,
                                bottom[0]->shape()[0],
                                this->channels_,
                                &(bottom[0]->shape()[2]));
  const int_tp* pooled_size_data = this->pooled_size_.cpu_data();
  cudnn::setTensorNdDesc<Dtype>(&top_desc_,
                                bottom[0]->shape().size() - 2,
                                bottom[0]->shape()[0],
                                this->channels_,
                                pooled_size_data);
}

template <typename Dtype>
CuDNNPoolingLayer<Dtype>::~CuDNNPoolingLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyPoolingDescriptor(pooling_desc_);
  cudnnDestroy(handle_);
}

INSTANTIATE_CLASS(CuDNNPoolingLayer);

}   // namespace caffe
#endif
