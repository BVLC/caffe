// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe {
template <typename Dtype>
void AccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  top_k_ = this->layer_param_.accuracy_param().top_k();
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
      << "top_k must be less than or equal to the number of classes.";
  CHECK_GE(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->channels() % bottom[1]->channels(), 0);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
Dtype AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int label_channels =  bottom[1]->channels();
  int data_chan_split = channels / bottom[1]->channels();;
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  for (int i = 0; i < num; ++i) {
    int intra_chan_accuracy = 0;
    int data_step = 0;
    for (int c = 0; c < label_channels; ++c) {
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int j = 0; j < data_chan_split; ++j) {
        bottom_data_vector.push_back(
            std::make_pair(bottom_data[i * channels + data_step + j], j));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
        // check if true label is in top k predictions
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second
            == static_cast<int>(bottom_label[i * label_channels + c ])) {
          ++intra_chan_accuracy;
          break;
        }
      }
      data_step += data_chan_split;
    }
    if (intra_chan_accuracy == label_channels) {
      ++accuracy;
    }
  }
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;

  // Accuracy layer should not be used as a loss function.
  return Dtype(0);
}

INSTANTIATE_CLASS(AccuracyLayer);

}  // namespace caffe
