#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_lrn_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNLRNLayer<Dtype, MItype, MOtype>::LayerSetUp(const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  LRNLayer<Dtype, MItype, MOtype>::LayerSetUp(bottom, top);

  CUDNN_CHECK(cudnnCreate(&handle_));
  CUDNN_CHECK(cudnnCreateLRNDescriptor(&norm_desc_));
  cudnn::createTensorNdDesc<Dtype>(&bottom_desc_);
  cudnn::createTensorNdDesc<Dtype>(&top_desc_);

  // create a LRN handle
  handles_setup_ = true;

  size_ = this->layer_param().lrn_param().local_size();
  alpha_ = this->layer_param().lrn_param().alpha();
  beta_ = this->layer_param().lrn_param().beta();
  k_ = this->layer_param().lrn_param().k();
}

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNLRNLayer<Dtype, MItype, MOtype>::Reshape(const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  LRNLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);

  vector<int_tp> shape;

  shape.push_back(bottom[0]->num());
  shape.push_back(this->channels_);
  shape.push_back(this->height_);
  shape.push_back(this->width_);


  const int_tp* shape_ptr = &shape[0];

  cudnn::setTensorNdDesc<Dtype>(&bottom_desc_, 4, shape_ptr);
  cudnn::setTensorNdDesc<Dtype>(&top_desc_, 4, shape_ptr);
  CUDNN_CHECK(cudnnSetLRNDescriptor(norm_desc_, size_, alpha_, beta_, k_));
}

template<typename Dtype, typename MItype, typename MOtype>
CuDNNLRNLayer<Dtype, MItype, MOtype>::~CuDNNLRNLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);

  // destroy LRN handle
  cudnnDestroy(handle_);
}

INSTANTIATE_CLASS_3T_GUARDED(CuDNNLRNLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(CuDNNLRNLayer, (double), (double), (double));

}   // namespace caffe
#endif
