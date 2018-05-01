#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_relu6_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNReLU6Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ReLU6Layer<Dtype>::LayerSetUp(bottom, top);
  // initialize cuDNN
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  ///cudnn::createActivationDescriptor<Dtype>(&activ_desc_, CUDNN_ACTIVATION_RELU);
  cudnn::createActivationDescriptor<Dtype>(&activ_desc_, CUDNN_ACTIVATION_CLIPPED_RELU);
  cudnnSetActivationDescriptor(activ_desc_, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_NOT_PROPAGATE_NAN, Dtype(6));
  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNReLU6Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ReLU6Layer<Dtype>::Reshape(bottom, top);
  const int N = bottom[0]->num();
  const int K = bottom[0]->channels();
  const int H = bottom[0]->height();
  const int W = bottom[0]->width();
  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, N, K, H, W);
  cudnn::setTensor4dDesc<Dtype>(&top_desc_, N, K, H, W);
}

template <typename Dtype>
CuDNNReLU6Layer<Dtype>::~CuDNNReLU6Layer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(this->bottom_desc_);
  cudnnDestroyTensorDescriptor(this->top_desc_);
  cudnnDestroyActivationDescriptor(this->activ_desc_);
  cudnnDestroy(this->handle_);
}

INSTANTIATE_CLASS(CuDNNReLU6Layer);

}  // namespace caffe
#endif
