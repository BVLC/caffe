#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(this->layer_param_.reshape_param().shape());
  top[0]->ShareData(*bottom[0]);
  top[0]->ShareDiff(*bottom[0]);
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top[0]->count(), bottom[0]->count())
     << "new shape must have the same count as input";
}

INSTANTIATE_CLASS(ReshapeLayer);
REGISTER_LAYER_CLASS(Reshape);

}  // namespace caffe
