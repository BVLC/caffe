#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NormalizedSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, &sigmoid_top_vec_);
}

template <typename Dtype>
void NormalizedSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "NORMALIZED_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, &sigmoid_top_vec_);
}

template <typename Dtype>
void NormalizedSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, &sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int dim = count / num;
  
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  
  for (int i = 0; i < dim; ++i) {
    int n_pos = 0;
    int n_neg = 0;
    Dtype pos_loss = 0;
    Dtype neg_loss = 0;
    for (int j = 0; j < num; ++j) {
      int idx = j * dim + i;
      if (target[idx] > 0.5) {
        n_pos++;
        pos_loss -= input_data[idx] * (target[idx] - (input_data[idx] >= 0)) -
        log(1 + exp(input_data[idx] - 2 * input_data[idx] * (input_data[idx] >= 0)));
      } else {
        n_neg++;
        neg_loss -= input_data[idx] * (target[idx] - (input_data[idx] >= 0)) -
        log(1 + exp(input_data[idx] - 2 * input_data[idx] * (input_data[idx] >= 0)));
      }
    }
    // Only count loss if there are both positive and negative samples
    if (n_pos > 0 && n_pos < num) {
      const float ratio = float(n_pos) / n_neg;
      // Only normalize if ratio reaches threshold
      if (ratio >= thres_ || 1. / ratio >= thres_ ) {
        loss += pos_loss / (n_pos * 2);
        loss += neg_loss / (n_neg * 2);
      } else {
        loss += pos_loss / num;
        loss += neg_loss / num;
      }
    }
  }
  (*top)[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void NormalizedSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = (*bottom)[0]->count();
    const int num = (*bottom)[0]->num();
    const int dim = count / num;
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = (*bottom)[1]->cpu_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    Dtype* scales = new Dtype[count];

    for (int i = 0; i < dim; ++i) {
      int n_pos = 0;
      int n_neg = 0;
      for (int j = 0; j < num; ++j) {
        int idx = j * dim + i;
        if (target[idx] > 0.5) {
          n_pos++;
        } else {
          n_neg++;
        }
      }
      // Only back propagate if there are both positive and negative samples
      if (n_pos > 0 && n_pos < num) {
        const float ratio = float(n_pos) / n_neg;
        const bool shouldNorm = (ratio >= thres_ || 1. / ratio >= thres_);
        for (int j = 0; j < num; ++j) {
          int idx = j * dim + i;
          if (target[idx] > 0.5) {
            if (shouldNorm) {
              scales[idx] = loss_weight / (n_pos * 2.);
            } else {
              scales[idx] = loss_weight / num;
            }
          } else {
            if (shouldNorm) {
              scales[idx] = loss_weight / (n_neg * 2.);
            } else {
              scales[idx] = loss_weight / num;
            }
          }
        }
      }
    }
    caffe_mul(count, scales, bottom_diff, bottom_diff);
    delete [] scales;
  }
}

#ifdef CPU_ONLY
STUB_GPU(NormalizedSigmoidCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(NormalizedSigmoidCrossEntropyLossLayer);


}  // namespace caffe
