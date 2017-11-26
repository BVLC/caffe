#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  PoolingLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  Reshape_const(bottom, top);
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Reshape_const(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) const {
  PoolingLayer<Dtype>::Reshape_const(bottom, top);
}

template <typename Dtype>
CuDNNPoolingLayer<Dtype>::~CuDNNPoolingLayer() = default;

INSTANTIATE_CLASS(CuDNNPoolingLayer);

} // namespace caffe
#endif
