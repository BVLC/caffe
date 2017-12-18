#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_tanh_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNTanHLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  TanHLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void CuDNNTanHLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Reshape_const(bottom, top);
}


template <typename Dtype>
void CuDNNTanHLayer<Dtype>::Reshape_const(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const {
  TanHLayer<Dtype>::Reshape_const(bottom, top);
}


INSTANTIATE_CLASS(CuDNNTanHLayer);

}  // namespace caffe
#endif
