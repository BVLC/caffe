// Copyright 2014 BVLC and contributors.

#include <cfloat>
#include <queue>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ArgMaxLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  out_max_val_ = this->layer_param_.argmax_param().out_max_val();
  top_k_ = this->layer_param_.argmax_param().top_k();
  CHECK_GE(top_k_, 1) << " top k must not be less than 1.";
  CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
      << "top_k must be less than or equal to the number of classes.";
  if (out_max_val_) {
    // Produces max_ind and max_val
    (*top)[0]->Reshape(bottom[0]->num(), 2, top_k_, 1);
  } else {
    // Produces only max_ind
    (*top)[0]->Reshape(bottom[0]->num(), 1, top_k_, 1);
  }
}

template <typename Dtype>
class IDAndValueComparator {
 public:
  bool operator() (const std::pair<size_t, Dtype>& lhs,
                   const std::pair<size_t, Dtype>& rhs) const {
    return lhs.second < rhs.second || (lhs.second == rhs.second &&
        lhs.first < rhs.first);
  }
};

template <typename Dtype>
Dtype ArgMaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype value;
  for (int i = 0; i < num; ++i) {
    std::priority_queue<std::pair<size_t, Dtype>,
        std::vector<std::pair<size_t, Dtype> >, IDAndValueComparator<Dtype> >
        top_k_results;
    for (int j = 0; j < dim; ++j) {
      value = -(bottom_data[i * dim + j]);
      if (top_k_results.size() >= top_k_) {
        if (value < top_k_results.top().second) {
          top_k_results.pop();
          top_k_results.push(std::make_pair(j, value));
        }
      } else {
        top_k_results.push(std::make_pair(j, value));
      }
    }
    if (out_max_val_) {
      for (int j = 0; j < top_k_; ++j) {
        top_data[i * 2 * top_k_ + (top_k_ - 1 - j) * 2] =
            top_k_results.top().first;
        top_data[i * 2 * top_k_ + (top_k_ - 1 - j) * 2 + 1] =
            -(top_k_results.top().second);
        top_k_results.pop();
      }
    } else {
      for (int j = 0; j < top_k_; ++j) {
        top_data[i * top_k_ + (top_k_ - 1 - j)] = top_k_results.top().first;
        top_k_results.pop();
      }
    }
  }
  return Dtype(0);
}

INSTANTIATE_CLASS(ArgMaxLayer);

}  // namespace caffe
