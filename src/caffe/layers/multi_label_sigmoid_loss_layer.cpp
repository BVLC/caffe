#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/multilabel_sigmoid_loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
void MultiLabelSigmoidLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

   LOG(INFO) << bottom[0]->num() << " " << bottom[0]->channels() << " " << bottom[0]->height() << " " << bottom[0]->width() ;
   LOG(INFO) << bottom[1]->num() << " " << bottom[1]->channels() << " " << bottom[1]->height() << " " << bottom[1]->width() ;

  // number of channels of bottom[0] has to be number of classes
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());

  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  if (top.size() >= 1) {
   // sigmoid cross entropy loss (averaged across batch)
    top[0]->Reshape(1, 1, 1, 1);
  }
  if (top.size() == 2) {
   // softmax output
    top[1]->ReshapeLike(*sigmoid_output_.get());
    top[1]->ShareData(*sigmoid_output_.get());
  }
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void MultiLabelSigmoidLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

//  LOG(ERROR)<< "batch size:"<< bottom[0]->num();//batch size, defined in the network
//  LOG(ERROR)<< "batch channel:"<< bottom[0]->channels();//feature length, 4096
//  LOG(ERROR)<< "batch total nodes:"<< bottom[0]->count();//channels*size

  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
/*
  for(int i = 0; i < bottom[0]->num(); i++){
	for( int j = 0; j < bottom[0]->channels(); j++){
		LOG(ERROR)<< "current batch data:"<< "["<< i << "]"<< "["<< j << "]"<<":"<< bottom[0]->cpu_data()[i*bottom[0]->channels()+j];
	}
  }

  for(int i = 0; i < bottom[1]->num(); i++){
	for( int j = 0; j < bottom[1]->channels(); j++){
		LOG(ERROR)<< "current batch target:"<< "["<< i << "]"<< "["<< j << "]"<<":"<< bottom[1]->cpu_data()[i*bottom[1]->channels()+j];
	}
  }
*/

  Dtype loss = 0;
  int valid_count = 0;
  for (int i = 0; i < count; ++i) {
    if (target[i] >= 0) {
    // Update the loss only if target[i] is not -1
      loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
      ++valid_count;
    }
  }
  if (top.size() >= 1) {
    // top[0]->mutable_cpu_data()[0] = loss / num;
    if (valid_count)
      top[0]->mutable_cpu_data()[0] = loss / valid_count;
    else
      top[0]->mutable_cpu_data()[0] = loss / num;
  }
}

template <typename Dtype>
void MultiLabelSigmoidLossLayer<Dtype>::Backward_cpu(
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
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int valid_count = 0;
    for (int i = 0; i < count; ++i) {
      if (target[i] >= 0) {
        bottom_diff[i] = sigmoid_output_data[i] - target[i];
        valid_count++;
      } else {
        bottom_diff[i] = 0;
      }
    }
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (valid_count)
      caffe_scal(count, loss_weight / num, bottom_diff);
    else
      caffe_scal(count, loss_weight / valid_count, bottom_diff);
  }
}

#ifdef CPU_ONLY
//STUB_GPU_BACKWARD(MultiLabelSigmoidLossLayer, Backward);
STUB_GPU(MultiLabelSigmoidLossLayer);
#endif

INSTANTIATE_CLASS(MultiLabelSigmoidLossLayer);
REGISTER_LAYER_CLASS(MultiLabelSigmoidLoss);

}  // namespace caffe
