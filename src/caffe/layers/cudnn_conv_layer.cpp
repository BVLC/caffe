#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/syncedmem.hpp"

namespace caffe {

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  // Set the indexing parameters.
  bias_offset_ = (this->num_output_ / this->group_);


  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  Reshape_const(bottom,top);
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape_const(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) const {
  ConvolutionLayer<Dtype>::Reshape_const(bottom, top);
  CHECK_EQ(2, this->num_spatial_axes_)
      << "CuDNNConvolution input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND convolution.";


}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

} // namespace caffe
#endif
