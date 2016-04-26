#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/softmax_cross_entropy_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(softmax_output_.get());
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SOFTMAX_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax outputs.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
//  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  const Dtype* softmax_output_data = softmax_top_vec_[0]->cpu_data();

  // First compute max of input data
  for (int i = 0; i < count; ++i) {
    //loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
    //    log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    if (target[i] > 0 ) {
      loss -= target[i] * (log(softmax_output_data[i]) - log(target[i]));
    }
  }

  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* softmax_output_data = softmax_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // Gradient is: target[i] - softmax_output_data[i]
    caffe_sub(count, softmax_output_data, target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SoftmaxCrossEntropyLoss);

}  // namespace caffe
