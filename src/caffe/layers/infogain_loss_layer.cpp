#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/infogain_loss_layer.hpp"
#include "caffe/util/io.hpp"  // for bolb reading of matrix H
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InfogainLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // internal softmax layer
  LayerParameter softmax_layer_param(this->layer_param_);
  SoftmaxParameter* softmax_param = softmax_layer_param.mutable_softmax_param();
  softmax_param->set_axis(this->layer_param_.infogain_loss_param().axis());
  softmax_layer_param.set_type("Softmax");
  softmax_layer_param.clear_loss_weight();
  softmax_layer_param.add_loss_weight(1);
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_layer_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  // ignore label
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  // normalization
  CHECK(!this->layer_param_.loss_param().has_normalize())
    << "normalize is deprecated. use \"normalization\"";
  normalization_ = this->layer_param_.loss_param().normalization();
  // matrix H
  if (bottom.size() < 3) {
    CHECK(this->layer_param_.infogain_loss_param().has_source())
        << "Infogain matrix source must be specified.";
    BlobProto blob_proto;
    ReadProtoFromBinaryFile(
      this->layer_param_.infogain_loss_param().source(), &blob_proto);
    infogain_.FromProto(blob_proto);
  }
}

template <typename Dtype>
void InfogainLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  infogain_axis_ =
    bottom[0]->CanonicalAxisIndex(
      this->layer_param_.infogain_loss_param().axis());
  outer_num_ = bottom[0]->count(0, infogain_axis_);
  inner_num_ = bottom[0]->count(infogain_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if infogain axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  num_labels_ = bottom[0]->shape(infogain_axis_);
  Blob<Dtype>* infogain = NULL;
  if (bottom.size() < 3) {
    infogain = &infogain_;
  } else {
    infogain = bottom[2];
  }
  CHECK_EQ(infogain->count(), num_labels_*num_labels_);
  sum_rows_H_.Reshape(vector<int>(1, num_labels_));
  if (bottom.size() == 2) {
    // H is provided as a parameter and will not change. sum rows once
    sum_rows_of_H(infogain);
  }
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
Dtype InfogainLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void InfogainLossLayer<Dtype>::sum_rows_of_H(const Blob<Dtype>* H) {
  CHECK_EQ(H->count(), num_labels_*num_labels_)
    << "H must be " << num_labels_ << "x" << num_labels_;
  const Dtype* infogain_mat = H->cpu_data();
  Dtype* sum = sum_rows_H_.mutable_cpu_data();
  for ( int row = 0; row < num_labels_ ; row++ ) {
    sum[row] = 0;
    for ( int col = 0; col < num_labels_ ; col++ ) {
      sum[row] += infogain_mat[row*num_labels_+col];
    }
  }
}

template <typename Dtype>
void InfogainLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* infogain_mat = NULL;
  if (bottom.size() < 3) {
    infogain_mat = infogain_.cpu_data();
  } else {
    infogain_mat = bottom[2]->cpu_data();
  }
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value =
        static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels_);
      for (int l = 0; l < num_labels_; l++) {
        loss -= infogain_mat[label_value * num_labels_ + l] *
          log(std::max(
                prob_data[i * inner_num_*num_labels_ + l * inner_num_ + j],
                Dtype(kLOG_THRESHOLD)));
      }
      ++count;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void InfogainLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    const Dtype* infogain_mat = NULL;
    if (bottom.size() < 3) {
      infogain_mat = infogain_.cpu_data();
    } else {
      infogain_mat = bottom[2]->cpu_data();
      // H is provided as a "bottom" and might change. sum rows every time.
      sum_rows_of_H(bottom[2]);
    }
    const Dtype* sum_rows_H = sum_rows_H_.cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int dim = bottom[0]->count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, num_labels_);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int l = 0; l < num_labels_; ++l) {
            bottom_diff[i * dim + l * inner_num_ + j] = 0;
          }
        } else {
          for (int l = 0; l < num_labels_; ++l) {
            bottom_diff[i * dim + l * inner_num_ + j] =
               prob_data[i*dim + l*inner_num_ + j]*sum_rows_H[label_value]
               - infogain_mat[label_value * num_labels_ + l];
          }
          ++count;
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(bottom[0]->count(), loss_weight, bottom_diff);
  }
}

INSTANTIATE_CLASS(InfogainLossLayer);
REGISTER_LAYER_CLASS(InfogainLoss);
}  // namespace caffe
