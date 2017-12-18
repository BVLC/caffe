#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNSigmoidLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SigmoidLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void CuDNNSigmoidLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Reshape_const(bottom, top);
}

template <typename Dtype>
void CuDNNSigmoidLayer<Dtype>::Reshape_const(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const {
  SigmoidLayer<Dtype>::Reshape_const(bottom, top);
}

INSTANTIATE_CLASS(CuDNNSigmoidLayer);

}  // namespace caffe
#endif
