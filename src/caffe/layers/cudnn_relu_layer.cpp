#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  ReLULayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void CuDNNReLULayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  Reshape_const(bottom, top);
}

template <typename Dtype>
void CuDNNReLULayer<Dtype>::Reshape_const(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) const {
  ReLULayer<Dtype>::Reshape_const(bottom, top);
}

INSTANTIATE_CLASS(CuDNNReLULayer);

} // namespace caffe
#endif
