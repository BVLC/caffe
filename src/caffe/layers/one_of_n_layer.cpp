#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void OneOfNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  output_n_ = this->layer_param_.one_of_n_param().output_n();
  
  CHECK_GE(output_n_, 2) << " output n must not be less than 2.";
}

template <typename Dtype>
void OneOfNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), output_n_, 1, bottom[0]->count(3, 4));
}

template <typename Dtype>
void OneOfNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count(3, 4);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      int label = bottom_data[i * dim + j];
      if (label < 0) {
        label = 0;
      }
      if (label >= output_n_) {
        label = output_n_ - 1;
      }
      for (int k = 0; k < output_n_; ++k) {
          top_data[i * dim * output_n_ + j + k * dim] = 0;
      } 
      top_data[i * dim * output_n_ + j + label * dim] = 1;
    }
  }
}

INSTANTIATE_CLASS(OneOfNLayer);
REGISTER_LAYER_CLASS(OneOfN);

}  // namespace caffe
