#include <vector>

#include "caffe/layers/pairwise_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void PairwiseLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  
  threshold_ = this->layer_param_.pairwise_param().threshold();
  loss_weight_ = this->layer_param_.loss_weight(0);
}

template <typename Dtype>
void PairwiseLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);
  
  outer_num_ = bottom[0]->count(0, 1);
  inner_num_ = bottom[0]->count(1);
  label_dim_ = bottom[1]->count(1);
  
  pairwise_sim_.Reshape(1, 1, outer_num_, outer_num_);
  loss_.Reshape(1, 1, outer_num_, outer_num_);
  temp_.Reshape(1, 1, outer_num_, outer_num_);
}

template <typename Dtype>
void PairwiseLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void PairwiseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(PairwiseLossLayer);
#endif

INSTANTIATE_CLASS(PairwiseLossLayer);
REGISTER_LAYER_CLASS(PairwiseLoss);

}  // namespace caffe
