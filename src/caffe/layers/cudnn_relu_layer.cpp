#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_relu_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNReLULayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  ReLULayer<Dtype, MItype, MOtype>::LayerSetUp(bottom, top);
  // initialize cuDNN
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensorNdDesc<Dtype>(&bottom_desc_);
  cudnn::createTensorNdDesc<Dtype>(&top_desc_);
  cudnn::createActivationDescriptor<Dtype>(&activ_desc_, CUDNN_ACTIVATION_RELU);
  handles_setup_ = true;
}

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNReLULayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  ReLULayer<Dtype, MItype, MOtype>::Reshape(bottom, top);
  cudnn::setTensorNdDesc<Dtype>(&bottom_desc_, bottom[0]->shape().size(),
                                &(bottom[0]->shape()[0]));
  cudnn::setTensorNdDesc<Dtype>(&top_desc_, top[0]->shape().size(),
                                &(top[0]->shape()[0]));
}

template<typename Dtype, typename MItype, typename MOtype>
CuDNNReLULayer<Dtype, MItype, MOtype>::~CuDNNReLULayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(this->bottom_desc_);
  cudnnDestroyTensorDescriptor(this->top_desc_);
  cudnnDestroyActivationDescriptor(this->activ_desc_);
  cudnnDestroy(this->handle_);
}

INSTANTIATE_CLASS_3T_GUARDED(CuDNNReLULayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(CuDNNReLULayer, (double), (double), (double));

}  // namespace caffe
#endif
