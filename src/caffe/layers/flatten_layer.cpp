#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void FlattenLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(2);
  top_shape[0] = bottom[0]->num();
  top_shape[1] = bottom[0]->count() / bottom[0]->num();
  top[0]->Reshape(top_shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count());
  top[0]->ShareData(*bottom[0]);
  top[0]->ShareDiff(*bottom[0]);
}

INSTANTIATE_CLASS(FlattenLayer);
REGISTER_LAYER_CLASS(Flatten);

}  // namespace caffe
