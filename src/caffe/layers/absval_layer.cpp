#include <vector>

#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void AbsValLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
}

template<typename Dtype>
void AbsValLayer<Dtype>::forward(const Caffe::Brew brew,
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_data(brew);
  GetDevice<Dtype>(brew)->abs(count, bottom[0]->data(brew), top_data);
}

template<typename Dtype>
void AbsValLayer<Dtype>::backward(const Caffe::Brew brew,
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->diff(brew);
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->data(brew);
    Dtype* bottom_diff = bottom[0]->mutable_diff(brew);
    GetDevice<Dtype>(brew)->sign(count, bottom_data, bottom_diff);
    GetDevice<Dtype>(brew)->mul(count, bottom_diff, top_diff, bottom_diff);
  }
}

INSTANTIATE_CLASS(AbsValLayer);
REGISTER_LAYER_CLASS(AbsVal);

}  // namespace caffe
