#ifdef USE_CUDNN
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/cudnn_softmax_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNSoftmaxLayer<Dtype, MItype, MOtype>::LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  SoftmaxLayer<Dtype, MItype, MOtype>::LayerSetUp(bottom, top);
  // Initialize CUDNN.
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensorNdDesc<Dtype>(&bottom_desc_);
  cudnn::createTensorNdDesc<Dtype>(&top_desc_);
  handles_setup_ = true;
}

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNSoftmaxLayer<Dtype, MItype, MOtype>::Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  SoftmaxLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);

  vector<int_tp> shape;

  shape.push_back(this->outer_num_);
  shape.push_back(bottom[0]->shape(this->softmax_axis_));
  shape.push_back(this->inner_num_);
  shape.push_back(1);


  const int_tp* shape_ptr = &shape[0];

  cudnn::setTensorNdDesc<Dtype>(&bottom_desc_, 4, shape_ptr);
  cudnn::setTensorNdDesc<Dtype>(&top_desc_, 4, shape_ptr);
}

template<typename Dtype, typename MItype, typename MOtype>
CuDNNSoftmaxLayer<Dtype, MItype, MOtype>::~CuDNNSoftmaxLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroy(handle_);
}

INSTANTIATE_CLASS_3T_GUARDED(CuDNNSoftmaxLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(CuDNNSoftmaxLayer, (double), (double), (double));


}  // namespace caffe
#endif
