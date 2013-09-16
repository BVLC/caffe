#include "caffeine/vision_layers.hpp"
#include <algorithm>

using std::max;

namespace caffeine {

template <typename Dtype>
void NeuronLayer<Dtype>::SetUp(vector<const Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Neuron Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Neuron Layer takes a single blob as output.";
  for (int i = 0; i < bottom.size(); ++i) {
    (*top)[i].Reshape(bottom.num(), bottom.channels(), bottom.height(),
                      bottom.width());
  }
};

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0].cpu_data();
  Dtype* top_data = (*top)[0].mutable_cpu_data();
  const int count = bottom[0].count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = max(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
Dtype ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0].cpu_data();
    const Dtype* top_diff = top[0].cpu_diff();
    Dtype* bottom_diff = (*bottom)[0].mutable_cpu_diff();
    const int count = (*bottom)[0].count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * (bottom_data[i] >= 0);
    }
  }
  return Dtype(0);
}

template <typename Dtype>
inline void ReLULayer<Dtype>::Predict_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Forward_cpu(bottom, top);
}


}  // namespace caffeine
