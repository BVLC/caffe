#include <algorithm>
#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WeightedLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (bottom.size() < 3) {
    CHECK(this->layer_param_.weighted_loss_param().has_source())
        << "WeightedLoss matrix source must be specified.";
    const string& weights_file =
        this->layer_param_.weighted_loss_param().source();
    LOG(INFO) << "WEIGHTED_LOSS: Loading weights from file: " << weights_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(weights_file.c_str(), &blob_proto);
    weights_.FromProto(blob_proto);
  }
}

template <typename Dtype>
void WeightedLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  Blob<Dtype>* weights = NULL;
  if (bottom.size() < 3) {
    weights = &weights_;
  } else {
    weights = bottom[2];
  }
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  CHECK_EQ(weights->num(), 1);
  CHECK_EQ(weights->channels(), 1);
  CHECK_EQ(weights->height(), dim);
  CHECK_EQ(weights->width(), dim);
}

template <typename Dtype>
void WeightedLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* weights_mat = NULL;
  if (bottom.size() < 3) {
    weights_mat = weights_.cpu_data();
  } else {
    weights_mat = bottom[2]->cpu_data();
  }
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    for (int j = 0; j < dim; ++j) {
      loss += weights_mat[label * dim + j] * bottom_data[i * dim + j];
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void WeightedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down.size() > 2 && propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to infogain inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_label = bottom[1]->cpu_data();
    const Dtype* weights_mat = NULL;
    if (bottom.size() < 3) {
      weights_mat = weights_.cpu_data();
    } else {
      weights_mat = bottom[2]->cpu_data();
    }
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    const Dtype scale = top[0]->cpu_diff()[0] / num;
    for (int i = 0; i < num; ++i) {
      const int label = static_cast<int>(bottom_label[i]);
      for (int j = 0; j < dim; ++j) {
        bottom_diff[i * dim + j] = scale * weights_mat[label * dim + j];
      }
    }
  }
}

INSTANTIATE_CLASS(WeightedLossLayer);
REGISTER_LAYER_CLASS(WeightedLoss);
}  // namespace caffe
