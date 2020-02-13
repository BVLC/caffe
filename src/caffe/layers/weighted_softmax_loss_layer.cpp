#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/weighted_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SoftmaxWithLossLayer<Dtype>::LayerSetUp(bottom, top);

  this->softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());

  // set up tile layer for fast backward computation
  LayerParameter tile_param;
  TileParameter* tile_layer_params = tile_param.mutable_tile_param();
  tile_layer_params->set_axis(this->softmax_axis_);
  tile_layer_params->set_tiles(bottom[0]->shape(this->softmax_axis_));
  tile_param.set_type("Tile");
  tile_layer_ = LayerRegistry<Dtype>::CreateLayer(tile_param);
  tile_bottom_vec_.clear();
  tile_bottom_vec_.push_back(bottom[2]);  // tile the weights
  tile_top_vec_.clear();
  tile_top_vec_.push_back(&tweight_);
  tile_layer_->SetUp(tile_bottom_vec_, tile_top_vec_);

  this->has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (this->has_ignore_label_) {
    this->ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    this->normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    this->normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SoftmaxWithLossLayer<Dtype>::Reshape(bottom, top);
  tile_layer_->Reshape(tile_bottom_vec_, tile_top_vec_);
  this->outer_num_ = bottom[0]->count(0, this->softmax_axis_);
  this->inner_num_ = bottom[0]->count(this->softmax_axis_ + 1);
  CHECK_EQ(this->outer_num_ * this->inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  CHECK_EQ(bottom[1]->count(), bottom[2]->count())
      << "Weights and labels must have the same shape";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  this->softmax_layer_->Forward(this->softmax_bottom_vec_,
                                this->softmax_top_vec_);
  const Dtype* prob_data = this->prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* weight = bottom[2]->cpu_data();
  int dim = this->prob_.count() / this->outer_num_;
  Dtype agg_weight = 0;
  Dtype loss = 0;
  for (int i = 0; i < this->outer_num_; ++i) {
    for (int j = 0; j < this->inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * this->inner_num_ + j]);
      const Dtype weight_value = weight[i * this->inner_num_ + j];
      if (this->has_ignore_label_ && label_value == this->ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, this->prob_.shape(this->softmax_axis_));
      DCHECK_GE(weight_value, 0);  // do not allow negative weights...
      loss -= weight_value
        * log(std::max(prob_data[i * dim + label_value * this->inner_num_ + j],
                           Dtype(FLT_MIN)));
      agg_weight += weight_value;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss
    / this->get_normalizer(this->normalization_, agg_weight);
  if (top.size() == 2) {
    top[1]->ShareData(this->prob_);
  }
}

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " LAyer cannot backpropagate to weight inputs.";
  }
  if (propagate_down[0]) {
    tile_layer_->Forward(tile_bottom_vec_, tile_top_vec_);
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = this->prob_.cpu_data();
    const Dtype* weight = bottom[2]->cpu_data();
    // gradient of all entries (apart from g.t. label) is the probability
    caffe_copy(this->prob_.count(), prob_data, bottom_diff);
    const Dtype* tweight = tweight_.cpu_data();
    // weighted probabilities as gradient baseline
    caffe_mul(tweight_.count(), bottom_diff, tweight, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = this->prob_.count() / this->outer_num_;
    Dtype agg_weight = 0;
    for (int i = 0; i < this->outer_num_; ++i) {
      for (int j = 0; j < this->inner_num_; ++j) {
        const int label_value =
          static_cast<int>(label[i * this->inner_num_ + j]);
        const Dtype weight_value = weight[i * this->inner_num_ + j];
        if (this->has_ignore_label_ && label_value == this->ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(this->softmax_axis_); ++c) {
            bottom_diff[i * dim + c * this->inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * this->inner_num_ + j]
            -= weight_value;
          agg_weight += weight_value;
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        this->get_normalizer(this->normalization_, agg_weight);
    caffe_scal(this->prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(WeightedSoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(WeightedSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(WeightedSoftmaxWithLoss);

}  // namespace caffe
