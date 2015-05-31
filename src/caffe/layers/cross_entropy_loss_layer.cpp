// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int count = bottom[0]->count(); 
  int num = bottom[0]->num(); 
  Dtype loss = 0; Dtype l;
  for (int i = 0; i < count; ++i) {
    if(bottom_label[i] != 0){
      l = -bottom_label[i]*log(max(bottom_data[i],Dtype(0.00001)));
      loss += l;
    }

    if(bottom_label[i] != 1){
      l = -(1-bottom_label[i])*log(max((1-bottom_data[i]),Dtype(0.00001)));
      loss += l;
    }
    CHECK_GE(l, 0.) << "loss is not >= 0, loss: " << l << " bottom_label: " << bottom_label[i] << " bottom_data: " << bottom_data[i];
  }
  top[0]->mutable_cpu_data()[0] = loss / count;
  // LOG(INFO) << "CrossEntropyLossLayer: " << loss / count;
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
      << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int count = bottom[0]->count();
    int num = bottom[0]->num();
    memset(bottom_diff, 0, sizeof(Dtype) * bottom[0]->count());
    for (int i = 0; i < count; ++i) {
      Dtype val = 0;
      if(bottom_label[i] != 0)
        val = val - bottom_label[i]/max(bottom_data[i], Dtype(0.00001))/count;
      if(bottom_label[i] != 1)
        val = val - (-(1-bottom_label[i])/max(1-bottom_data[i], Dtype(0.00001))/count);

      bottom_diff[i] = val;
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(CrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(CrossEntropyLossLayer);
REGISTER_LAYER_CLASS(CrossEntropyLoss);

}  // namespace caffe
