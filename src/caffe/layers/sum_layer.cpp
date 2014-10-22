#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void SumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                 vector<Blob<Dtype>*>* top) {
  (*top)[0]->Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void SumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                     vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / bottom[0]->num();
  for (int i = 0; i < num; ++i) {
    Dtype sum = 0;
    for (int j = 0; j < dim; ++j) {
      sum += bottom_data[i * dim + j];
    }
    top_data[i] = sum;
  }
}

INSTANTIATE_CLASS(SumLayer);

}  // namespace caffe
