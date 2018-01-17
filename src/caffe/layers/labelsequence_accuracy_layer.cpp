#include <functional>
#include <utility>
#include <vector>
//#include <sstream>
//#include <iterator>
#include "caffe/layers/labelsequence_accuracy_layer.hpp"

namespace caffe {

template <typename Dtype>
void LabelsequenceAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  blank_label_ = this->layer_param_.labelsequence_accuracy_param().blank_label();
}

template <typename Dtype>
void LabelsequenceAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void LabelsequenceAccuracyLayer<Dtype>::GetLabelseqs(
  const vector<int>& label_seq_with_blank,
  vector<int>& label_seq) {
  label_seq.clear();
  int prev = blank_label_;
  int length = label_seq_with_blank.size();
  for(int i = 0; i < length; ++i) {
    int cur = label_seq_with_blank[i];
    if(cur != prev && cur != blank_label_) {
      label_seq.push_back(cur);
    }
    prev = cur;
  }
}

static bool hit(const vector<int>& prediction, const vector<int>& gt) {
  if(prediction.size() != gt.size())  return false;
  for(int i = 0; i < gt.size(); ++i) {
    if(prediction[i] != gt[i])  return false;
  }
  return true;
}

template <typename Dtype>
void LabelsequenceAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // bottom[0]: NxTxC
  // bottom[1]: NxL
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0)) << "The batch size shoule be equal.";
  int cnt = 0;
  const Dtype* pred_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  int time_step = bottom[0]->shape(1);
  int alphabet_size = bottom[0]->shape(2);
  int label_seq_length = bottom[1]->shape(1);
  vector<int> pred_label_seq_with_blank(time_step);
  for(int n = 0; n < bottom[0]->shape()[0]; ++n) {
    for(int t = 0; t < time_step; ++t) {
      pred_label_seq_with_blank[t] = std::max_element(pred_data, pred_data + alphabet_size) - pred_data;
      pred_data += alphabet_size;
    }
    GetLabelseqs(pred_label_seq_with_blank, pred_label_seq_);
    gt_label_seq_.clear();
    for(int i = 0; i < label_seq_length; ++i) {
      if(gt_data[i] != blank_label_) {
        gt_label_seq_.push_back(gt_data[i]);
      }
    }
    gt_data += label_seq_length;
    //ostringstream ss;
    //std::copy(pred_label_seq_.begin(), pred_label_seq_.end(), std::ostream_iterator<Dtype>(ss, ", "));
    //ss << "\n***************\n";
    //std::copy(gt_label_seq_.begin(), gt_label_seq_.end(), std::ostream_iterator<Dtype>(ss, ", "));
    //ss << "\n--------------\n";
    //LOG(INFO) << ss.str();
    if(hit(pred_label_seq_, gt_label_seq_)) {
      ++cnt;
    }
  }
  Dtype accuracy = static_cast<Dtype>(cnt) / bottom[0]->shape()[0];
  top[0]->mutable_cpu_data()[0] = accuracy;
}

INSTANTIATE_CLASS(LabelsequenceAccuracyLayer);
REGISTER_LAYER_CLASS(LabelsequenceAccuracy);

}  // namespace caffe
