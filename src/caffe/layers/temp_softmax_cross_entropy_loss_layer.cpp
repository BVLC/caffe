#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TempSoftmaxCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  temperature = this->layer_param_.tempsoftmax_param().temperature();
  CHECK_GT(temperature, 1)
      << "Gradient assumes a (high) softmax temperature of greater than 1.";

  LayerParameter mvn_param(this->layer_param_);
  mvn_param.set_type("MVN");
  mvn_param.mutable_mvn_param()->set_normalize_variance(false);
  mvn_param.mutable_mvn_param()->set_across_channels(true);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");

  // network predictions, zero-mean and softmax
  mvn_in_layer_ = LayerRegistry<Dtype>::CreateLayer(mvn_param);
  mvn_in_bottom_vec_.clear();
  mvn_in_bottom_vec_.push_back(bottom[0]);
  mvn_in_top_vec_.clear();
  mvn_in_top_vec_.push_back(&mvn_in_output_);
  mvn_in_layer_->SetUp(mvn_in_bottom_vec_, mvn_in_top_vec_);

  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(mvn_in_top_vec_[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  // target, zero-mean and softmax
  mvn_target_layer_ = LayerRegistry<Dtype>::CreateLayer(mvn_param);
  mvn_target_bottom_vec_.clear();
  mvn_target_bottom_vec_.push_back(bottom[1]);
  mvn_target_top_vec_.clear();
  mvn_target_top_vec_.push_back(&mvn_target_output_);
  mvn_target_layer_->SetUp(mvn_target_bottom_vec_, mvn_target_top_vec_);

  softmax_target_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_target_bottom_vec_.clear();
  softmax_target_bottom_vec_.push_back(mvn_target_top_vec_[0]);
  softmax_target_top_vec_.clear();
  softmax_target_top_vec_.push_back(&target_prob_);
  softmax_target_layer_->SetUp(softmax_target_bottom_vec_,
          softmax_target_top_vec_);
}

template <typename Dtype>
void TempSoftmaxCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "TEMP_SOFTMAX_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  mvn_in_layer_->Reshape(mvn_in_bottom_vec_, mvn_in_top_vec_);
  mvn_target_layer_->Reshape(mvn_target_bottom_vec_, mvn_target_top_vec_);
}

template <typename Dtype>
void TempSoftmaxCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass mean normalises the bottom's.
  mvn_in_bottom_vec_[0] = bottom[0];
  mvn_in_layer_->Forward(mvn_in_bottom_vec_, mvn_in_top_vec_);
  mvn_target_bottom_vec_[0] = bottom[1];
  mvn_target_layer_->Forward(mvn_target_bottom_vec_, mvn_target_top_vec_);
  // Divide by temperature
  const int count = bottom[0]->count();
  softmax_bottom_vec_[0] = &mvn_in_output_;
  caffe_scal(count, Dtype(1) / temperature,
                    softmax_bottom_vec_[0]->mutable_cpu_data());

  softmax_target_bottom_vec_[0] = &mvn_target_output_;
  caffe_scal(count, Dtype(1) / temperature,
                    softmax_target_bottom_vec_[0]->mutable_cpu_data());
  // The forward pass computes the softmax values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  softmax_target_layer_->Forward(softmax_target_bottom_vec_,
                    softmax_target_top_vec_);

  const Dtype* input_data = prob_.cpu_data();
  const Dtype* target = target_prob_.cpu_data();

  // Compute the loss
  const int num = bottom[0]->num();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    loss -= target[i] * log(std::max(input_data[i], Dtype(FLT_MIN)));
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void TempSoftmaxCrossEntropyLossLayer<Dtype>::Backward_cpu(
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
    const Dtype* input_data = prob_.cpu_data();
    const Dtype* target = target_prob_.cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, input_data, target, bottom_diff);
    const int N = bottom[0]->channels();
    caffe_scal(count, Dtype(1) / (N * temperature * temperature), bottom_diff);

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(TempSoftmaxCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(TempSoftmaxCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(TempSoftmaxCrossEntropyLoss);

}  // namespace caffe
