#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"

#include "caffe/proto/reshape_param.pb.h"

namespace caffe {

template <typename Dtype>
void ReshapeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ReshapeParameter& param =
      this->layer_param_.GetExtension(reshape_param);
  top[0]->Reshape(param.shape());
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top[0]->count(), bottom[0]->count())
      << "New shape must have the same count as input shape.";
  top[0]->ShareData(*bottom[0]);
  top[0]->ShareDiff(*bottom[0]);
}

INSTANTIATE_CLASS(ReshapeLayer);
REGISTER_LAYER_CLASS(Reshape);

}  // namespace caffe
