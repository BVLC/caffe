#ifdef USE_CUDNN
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/cudnn_softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  SoftmaxLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  CuDNNSoftmaxLayer<Dtype>::Reshape_const(bottom, top);
}

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Reshape_const(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) const {
  SoftmaxLayer<Dtype>::Reshape_const(bottom, top);
}

template <typename Dtype>
CuDNNSoftmaxLayer<Dtype>::~CuDNNSoftmaxLayer() = default;

INSTANTIATE_CLASS(CuDNNSoftmaxLayer);

} // namespace caffe
#endif
