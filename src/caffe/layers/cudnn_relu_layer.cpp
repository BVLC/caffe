#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ReLULayer<Dtype>::LayerSetUp(bottom, top);
  // initialize cuDNN
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensorDesc<Dtype>(&bottom_desc_);
  cudnn::createTensorDesc<Dtype>(&top_desc_);
  cudnn::createActivationDescriptor<Dtype>(&activ_desc_, CUDNN_ACTIVATION_RELU);
  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ReLULayer<Dtype>::Reshape(bottom, top);
  cudnn::setTensorNdDesc<Dtype>(&bottom_desc_, bottom[0]->shape());
  cudnn::setTensorNdDesc<Dtype>(&top_desc_, bottom[0]->shape());
}

template <typename Dtype>
CuDNNReLULayer<Dtype>::~CuDNNReLULayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(this->bottom_desc_);
  cudnnDestroyTensorDescriptor(this->top_desc_);
  cudnnDestroyActivationDescriptor(this->activ_desc_);
  cudnnDestroy(this->handle_);
}

INSTANTIATE_CLASS(CuDNNReLULayer);

}  // namespace caffe
#endif
