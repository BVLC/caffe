#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/multibox_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
    this->layer_param_.add_propagate_down(false);
  }
  const MultiBoxLossParameter& multibox_loss_param =
      this->layer_param_.multibox_loss_param();

  num_ = bottom[0]->num();
  num_priors_ = bottom[2]->height() / 4;
  // Get other parameters.
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  num_classes_ = multibox_loss_param.num_classes();
  CHECK_GE(num_classes_, 1) << "num_classes must not be less than 1.";
  share_location_ = multibox_loss_param.share_location();
  loc_classes_ = share_location_ ? 1 : num_classes_;
  match_type_ = multibox_loss_param.match_type();
  overlap_threshold_ = multibox_loss_param.overlap_threshold();
  use_prior_for_matching_ = multibox_loss_param.use_prior_for_matching();
  background_label_id_ = multibox_loss_param.background_label_id();

  vector<int> loss_shape(1, 1);
  // Set up localization loss layer.
  loc_weight_ = multibox_loss_param.loc_weight();
  loc_loss_type_ = multibox_loss_param.loc_loss_type();
  // fake shape.
  vector<int> loc_shape(1, 4);
  loc_pred_.Reshape(loc_shape);
  loc_gt_.Reshape(loc_shape);
  loc_bottom_vec_.push_back(&loc_pred_);
  loc_bottom_vec_.push_back(&loc_gt_);
  loc_loss_.Reshape(loss_shape);
  loc_top_vec_.push_back(&loc_loss_);
  if (loc_loss_type_ == MultiBoxLossParameter_LocLossType_L2) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_l2_loc");
    layer_param.set_type("EuclideanLoss");
    layer_param.add_loss_weight(loc_weight_);
    loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
  } else {
    LOG(FATAL) << "Unknown localization loss type.";
  }
  // Set up confidence loss layer.
  conf_loss_type_ = multibox_loss_param.conf_loss_type();
  conf_bottom_vec_.push_back(&conf_pred_);
  conf_bottom_vec_.push_back(&conf_gt_);
  conf_loss_.Reshape(loss_shape);
  conf_top_vec_.push_back(&conf_loss_);
  if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_conf");
    layer_param.set_type("SoftmaxWithLoss");
    layer_param.add_loss_weight(Dtype(1.));
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    vector<int> conf_shape(1);
    // If we do not need to consider background, assume there is one match to
    // setup SoftmaxWithLossLayer correctly.
    conf_shape[0] = background_label_id_ > -1 ? num_ * num_priors_ : 1;
    conf_gt_.Reshape(conf_shape);
    conf_shape.push_back(num_classes_);
    conf_pred_.Reshape(conf_shape);
    conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    // Use SetUp() instead of LayerSetUp().
    conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  num_priors_ = bottom[2]->height() / 4;
  num_gt_ = bottom[3]->height();
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(num_priors_ * loc_classes_ * 4, bottom[0]->channels())
      << "Number of priors must match number of location predictions.";
  CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
      << "Number of priors must match number of confidence predictions.";
  vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* conf_data = bottom[1]->cpu_data();
  const Dtype* prior_data = bottom[2]->cpu_data();
  const Dtype* gt_data = bottom[3]->cpu_data();

  // Retrieve all ground truth.
  map<int, vector<NormalizedBBox> > all_gt_bboxes;
  GetGroundTruth(gt_data, num_gt_, background_label_id_, &all_gt_bboxes);

  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension.
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

  // Retrieve all predictions.
  vector<LabelBBox> all_loc_preds;
  GetLocPredictions(loc_data, num_, num_priors_, loc_classes_, share_location_,
                    &all_loc_preds);

  int num_matches = 0;
  for (int i = 0; i < num_; ++i) {
    map<int, vector<int> > match_indices;
    map<int, vector<float> > match_overlaps;
    // Check if there is ground truth for current image.
    if (all_gt_bboxes.find(i) == all_gt_bboxes.end()) {
      // There is no gt for current image. All predictions are negative.
      all_match_indices_.push_back(match_indices);
      all_match_overlaps_.push_back(match_overlaps);
      continue;
    }
    // Find match between predictions and ground truth.
    const vector<NormalizedBBox>& gt_bboxes = all_gt_bboxes.find(i)->second;
    for (int c = 0; c < loc_classes_; ++c) {
      int label = share_location_ ? -1 : c;
      if (!use_prior_for_matching_) {
        // Decode the prediction into bbox first.
        vector<NormalizedBBox> loc_bboxes;
        DecodeBBoxes(prior_bboxes, prior_variances, all_loc_preds[i][label],
                     &loc_bboxes);
        MatchBBox(gt_bboxes, loc_bboxes, label, match_type_,
                  overlap_threshold_, &match_indices[label],
                  &match_overlaps[label]);
      } else {
        MatchBBox(gt_bboxes, prior_bboxes, label, match_type_,
                  overlap_threshold_, &match_indices[label],
                  &match_overlaps[label]);
      }
      for (int m = 0; m < match_indices[label].size(); ++m) {
        if (match_indices[label][m] != -1) {
          num_matches++;
        }
      }
    }
    all_match_indices_.push_back(match_indices);
    all_match_overlaps_.push_back(match_overlaps);
  }

  if (num_matches >= 1) {
    // Form data to pass on to loc_loss_layer_.
    vector<int> loc_shape(2);
    loc_shape[0] = num_matches;
    loc_shape[1] = 4;
    loc_pred_.Reshape(loc_shape);
    loc_gt_.Reshape(loc_shape);
    Dtype* loc_pred_data = loc_pred_.mutable_cpu_data();
    Dtype* loc_gt_data = loc_gt_.mutable_cpu_data();
    int count = 0;
    for (int i = 0; i < num_; ++i) {
      if (all_gt_bboxes.find(i) == all_gt_bboxes.end()) {
        continue;
      }
      const map<int, vector<int> >& match_indices = all_match_indices_[i];
      for (map<int, vector<int> >::const_iterator it = match_indices.begin();
           it != match_indices.end(); ++it) {
        const int label = it->first;
        const vector<int>& match_index = it->second;
        const vector<NormalizedBBox>& loc_pred = all_loc_preds[i][label];
        CHECK_EQ(match_index.size(), loc_pred.size());
        CHECK_EQ(match_index.size(), prior_bboxes.size());
        for (int j = 0; j < match_index.size(); ++j) {
          if (match_index[j] == -1) {
            continue;
          }
          // Store location prediction.
          loc_pred_data[count * 4] = loc_pred[j].xmin();
          loc_pred_data[count * 4 + 1] = loc_pred[j].ymin();
          loc_pred_data[count * 4 + 2] = loc_pred[j].xmax();
          loc_pred_data[count * 4 + 3] = loc_pred[j].ymax();
          // Store encoded ground truth.
          const int gt_idx = match_index[j];
          CHECK_GT(all_gt_bboxes[i].size(), gt_idx);
          const NormalizedBBox& gt_bbox = all_gt_bboxes[i][gt_idx];
          NormalizedBBox gt_encode;
          EncodeBBox(prior_bboxes[j], prior_variances[j], gt_bbox, &gt_encode);
          loc_gt_data[count * 4] = gt_encode.xmin();
          loc_gt_data[count * 4 + 1] = gt_encode.ymin();
          loc_gt_data[count * 4 + 2] = gt_encode.xmax();
          loc_gt_data[count * 4 + 3] = gt_encode.ymax();
          count++;
        }
      }
    }
    loc_loss_layer_->Reshape(loc_bottom_vec_, loc_top_vec_);
    loc_loss_layer_->Forward(loc_bottom_vec_, loc_top_vec_);
  }

  // Form data to pass on to conf_loss_layer_.
  vector<int> conf_shape(1);
  conf_shape[0] = background_label_id_ > -1 ? num_ * num_priors_ : num_matches;
  conf_gt_.Reshape(conf_shape);
  conf_shape.push_back(num_classes_);
  conf_pred_.Reshape(conf_shape);
  Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();
  Dtype* conf_gt_data = conf_gt_.mutable_cpu_data();
  if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    if (background_label_id_ > -1) {
      // Need to consider background.
      // Directory copy the confidence prediction (but with different shape).
      CHECK_EQ(conf_pred_.count(), bottom[1]->count());
      caffe_copy(bottom[1]->count(), conf_data, conf_pred_.mutable_cpu_data());
      caffe_set(conf_gt_.count(), Dtype(background_label_id_), conf_gt_data);
    }
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
  int count = 0;
  for (int i = 0; i < num_; ++i) {
    if (all_gt_bboxes.find(i) != all_gt_bboxes.end()) {
      const map<int, vector<int> >& match_indices = all_match_indices_[i];
      for (int j = 0; j < num_priors_; ++j) {
        for (map<int, vector<int> >::const_iterator it = match_indices.begin();
             it != match_indices.end(); ++it) {
          const vector<int>& match_index = it->second;
          CHECK_EQ(match_index.size(), num_priors_);
          if (match_index[j] == -1) {
            continue;
          }
          const int gt_idx = match_index[j];
          int idx = background_label_id_ > -1 ? j : count;
          if (background_label_id_ == -1) {
            // Only copy scores for matched bboxes.
            for (int c = 0; c < num_classes_; ++c) {
              conf_pred_data[idx*num_classes_ + c] =
                  conf_data[j*num_classes_ + c];
            }
          }
          conf_gt_data[idx] = all_gt_bboxes[i][gt_idx].label();
          ++count;
        }
      }
    }
    if (background_label_id_ > -1) {
      conf_gt_data += num_priors_;
      conf_data += bottom[1]->offset(1);
    }
  }
  conf_loss_layer_->Reshape(conf_bottom_vec_, conf_top_vec_);
  conf_loss_layer_->Forward(conf_bottom_vec_, conf_top_vec_);

  top[0]->mutable_cpu_data()[0] = 0;
  if (this->layer_param_.propagate_down(0)) {
    // TODO(weiliu89): Understand why it needs to divide 2.
    top[0]->mutable_cpu_data()[0] += loc_weight_ * loc_loss_.cpu_data()[0] / 2;
  }
  if (this->layer_param_.propagate_down(1)) {
    // TODO(weiliu89): Understand why it needs to divide 2.
    top[0]->mutable_cpu_data()[0] += conf_loss_.cpu_data()[0] / 2;
  }
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to prior inputs.";
  }
  if (propagate_down[3]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }

  // Back propagate on location prediction.
  if (propagate_down[0]) {
    vector<bool> loc_propagate_down;
    // Only back propagate on prediction, not ground truth.
    loc_propagate_down.push_back(true);
    loc_propagate_down.push_back(false);
    loc_loss_layer_->Backward(loc_top_vec_, loc_propagate_down,
                              loc_bottom_vec_);
    const Dtype* loc_pred_diff = loc_pred_.cpu_diff();
    Dtype* loc_bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), loc_bottom_diff);
    int count = 0;
    for (int i = 0; i < num_; ++i) {
      const map<int, vector<int> >& match_indices = all_match_indices_[i];
      for (map<int, vector<int> >::const_iterator it = match_indices.begin();
           it != match_indices.end(); ++it) {
        const int label = share_location_ ? 0 : it->first;
        const vector<int>& match_index = it->second;
        for (int j = 0; j < match_index.size(); ++j) {
          if (match_index[j] == -1) {
            continue;
          }
          // Copy the diff to the right place.
          int start_idx = loc_classes_ * 4 * j + label * 4;
          for (int k = 0; k < 4; ++k) {
            loc_bottom_diff[start_idx + k] = loc_pred_diff[count * 4 + k];
          }
          count++;
        }
      }
      loc_bottom_diff += bottom[0]->offset(1);
    }
  }

  // Back propagate on confidence prediction.
  if (propagate_down[1]) {
    vector<bool> conf_propagate_down;
    // Only back propagate on prediction, not ground truth.
    conf_propagate_down.push_back(true);
    conf_propagate_down.push_back(false);
    conf_loss_layer_->Backward(conf_top_vec_, conf_propagate_down,
                               conf_bottom_vec_);
    Dtype* conf_bottom_diff = bottom[1]->mutable_cpu_diff();
    caffe_set(bottom[1]->count(), Dtype(0), conf_bottom_diff);
    const Dtype* conf_pred_diff = conf_pred_.cpu_diff();
    if (background_label_id_ == -1) {
      int count = 0;
      for (int i = 0; i < num_; ++i) {
        map<int, vector<int> >& match_indices = all_match_indices_[i];
        for (int j = 0; j < num_priors_; ++j) {
          for (map<int, vector<int> >::iterator it = match_indices.begin();
               it != match_indices.end(); ++it) {
            const vector<int>& match_index = it->second;
            CHECK_EQ(match_index.size(), num_priors_);
            if (match_index[j] == -1) {
              continue;
            }
            // Copy the diff to the right place.
            for (int c = 0; c < num_classes_; ++c) {
              conf_bottom_diff[j*num_classes_ + c] =
                  conf_pred_diff[count*num_classes_ + c];
            }
            ++count;
          }
        }
        conf_bottom_diff += bottom[1]->offset(1);
      }
    } else {
      // Copy the whole diff back.
      caffe_copy(conf_pred_.count(), conf_pred_diff, conf_bottom_diff);
    }
  }

  // After backward, remove match statistics.
  all_match_indices_.clear();
  all_match_overlaps_.clear();
}


#ifdef CPU_ONLY
STUB_GPU(MultiBoxLossLayer);
#endif

INSTANTIATE_CLASS(MultiBoxLossLayer);
REGISTER_LAYER_CLASS(MultiBoxLoss);

}  // namespace caffe
