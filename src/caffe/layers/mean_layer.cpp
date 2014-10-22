#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void MeanLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                 vector<Blob<Dtype>*>* top) {
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void MeanLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                     vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int count = bottom[0]->count();
  Dtype sum = 0;
  for (int j = 0; j < bottom[0]->count(); ++j) {
    sum += bottom_data[j];
  }
  (*top)[0]->mutable_cpu_data()[0] = sum / count;
}

INSTANTIATE_CLASS(MeanLayer);

}  // namespace caffe
