#include <algorithm>
#include <climits>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/hungarian.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void HungarianLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  const vector<int> target_confidence_shape(4);
  top[1]->Reshape(bottom[2]->shape());
  top[2]->Reshape(bottom[2]->shape());
  match_ratio_ = this->layer_param_.hungarian_loss_param().match_ratio();
  if (this->layer_param_.loss_weight_size() == 1) {
    this->layer_param_.add_loss_weight(Dtype(0));
    this->layer_param_.add_loss_weight(Dtype(0));
  }
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[1]->num(), bottom[2]->num());
}

template <typename Dtype>
void HungarianLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* box_pred_data = bottom[0]->cpu_data();
  const Dtype* boxes_data = bottom[1]->cpu_data();
  const Dtype* box_flags_data = bottom[2]->cpu_data();
  Dtype loss = 0.;
  assignments_.clear();
  num_gt_.clear();
  Dtype* top_confidences = top[1]->mutable_cpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    const int offset =
        n * bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
    num_gt_.push_back(0);
    for (int i = 0; i < bottom[2]->height(); ++i) {
      Dtype box_score = (*(box_flags_data + bottom[2]->offset(n, 0, i)));
      CHECK_NEAR(static_cast<int>(box_score), box_score, 0.01);
      num_gt_[n] += static_cast<int>(box_score);
    }
    const int channels = bottom[0]->channels();
    CHECK_EQ(channels, 4);

    const int num_pred = bottom[0]->height();

    vector<float> match_cost;
    vector<float> loss_mat;
    const int height = bottom[0]->height();
    for (int i = 0; i < num_pred; ++i) {
      for (int j = 0; j < num_pred; ++j) {
        const int idx = i * num_pred + j;
        match_cost.push_back(0.);
        loss_mat.push_back(0.);
        if (j >= num_gt_[n]) { continue; }
        for (int c = 0; c < channels; ++c) {
          const Dtype pred_value = box_pred_data[offset + c * height + i];
          const Dtype label_value = boxes_data[offset + c * height + j];
          match_cost[idx] += fabs(pred_value - label_value) / 2000.;
          loss_mat[idx] += fabs(pred_value - label_value);
        }
        CHECK_LT(match_cost[idx], 0.9);
        match_cost[idx] += i;
        const int c_x = 0;
        const int c_y = 1;
        const int c_w = 2;
        const int c_h = 3;

        const Dtype pred_x = box_pred_data[offset + c_x * num_pred + i];
        const Dtype pred_y = box_pred_data[offset + c_y * num_pred + i];

        const Dtype label_x = boxes_data[offset + c_x * num_pred + j];
        const Dtype label_y = boxes_data[offset + c_y * num_pred + j];
        const Dtype label_w = boxes_data[offset + c_w * num_pred + j];
        const Dtype label_h = boxes_data[offset + c_h * num_pred + j];

        float ratio;
        if (this->phase_ == TRAIN) {
          ratio = match_ratio_;
        } else {
          ratio = 1.0;
        }

        if (fabs(pred_x - label_x) / label_w > ratio ||
            fabs(pred_y - label_y) / label_h > ratio) {
          match_cost[idx] += 100;
        }
      }
    }


    double max_pair_cost = 0;
    for (int i = 0; i < num_pred; ++i) {
      for (int j = 0; j < num_pred; ++j) {
        const int idx = i * num_pred + j;
        max_pair_cost = std::max(max_pair_cost, fabs(match_cost[idx]));
      }
    }
    const int kMaxNumPred = 20;
    CHECK_LE(num_pred, kMaxNumPred);
    int int_cost[kMaxNumPred * kMaxNumPred];
    for (int i = 0; i < num_pred; ++i) {
      for (int j = 0; j < num_pred; ++j) {
        const int idx = i * num_pred + j;
        int_cost[idx] = static_cast<int>(
            match_cost[idx] / max_pair_cost * Dtype(INT_MAX) / 2.);
      }
    }

    std::vector<int> assignment;

    if (this->layer_param_.hungarian_loss_param().permute_matches()) {
      hungarian_problem_t p;
      int** m = array_to_matrix(int_cost, num_pred, num_pred);
      hungarian_init(&p, m, num_pred, num_pred, HUNGARIAN_MODE_MINIMIZE_COST);
      hungarian_solve(&p);
      for (int i = 0; i < num_pred; ++i) {
        for (int j = 0; j < num_pred; ++j) {
          if (p.assignment[i][j] == HUNGARIAN_ASSIGNED) {
            assignment.push_back(j);
          }
        }
      }
      CHECK_EQ(assignment.size(), num_pred);
      hungarian_free(&p);
      for (int i = 0; i < num_pred; ++i) {
        free(m[i]);
      }
      free(m);
    } else {
      for (int i = 0; i < height; ++i) {
        assignment.push_back(i);
      }
    }
    for (int i = 0; i < num_pred; ++i) {
      const int idx = i * num_pred + assignment[i];
      loss += loss_mat[idx];
    }
    assignments_.push_back(assignment);

    for (int i = 0; i < num_pred; ++i) {
      top_confidences[n * num_pred + i] =
          assignment[i] < num_gt_[n] ? Dtype(1) : Dtype(0);
      Dtype* top_assignments = top[2]->mutable_cpu_data();
      top_assignments[n * num_pred + i] =
          assignment[i] < num_gt_[n] ? assignment[i] : Dtype(-1);
    }
  }
  Dtype batch_size = bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss / batch_size;
}

template <typename Dtype>
void HungarianLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* box_pred_data = bottom[0]->cpu_data();
  const Dtype* boxes_data = bottom[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();

  Dtype* box_pred_diff = bottom[0]->mutable_cpu_diff();

  caffe_set(bottom[0]->count(), Dtype(0), box_pred_diff);

  Dtype batch_size = bottom[0]->num();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    const int offset =
        n * bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    CHECK_EQ(channels, 4);
    for (int i = 0; i < assignments_[n].size(); ++i) {
      const int j = assignments_[n][i];
      if (j >= num_gt_[n]) { continue; }
      for (int c = 0; c < channels; ++c) {
        const Dtype pred_value = box_pred_data[offset + c * height + i];
        const Dtype label_value = boxes_data[offset + c * height + j];
        box_pred_diff[offset + c * height + i] = top_diff[0] * (
            pred_value > label_value ? Dtype(1.) : Dtype(-1.)) / batch_size;
      }
    }
  }
}

INSTANTIATE_CLASS(HungarianLossLayer);
REGISTER_LAYER_CLASS(HungarianLoss);

}  // namespace caffe
