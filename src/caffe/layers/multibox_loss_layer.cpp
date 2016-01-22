#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/multibox_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::GetGroundTruth(const Dtype* gt_data,
      map<int, vector<NormalizedBBox> >* all_gt_bboxes) {
  for (int i = 0; i < num_gt_; ++i) {
    int start_idx = i * 7;
    int item_id = gt_data[start_idx];
    if (item_id == -1) {
      break;
    }
    NormalizedBBox bbox;
    bbox.set_label(gt_data[start_idx + 1]);
    bbox.set_xmin(gt_data[start_idx + 3]);
    bbox.set_ymin(gt_data[start_idx + 4]);
    bbox.set_xmax(gt_data[start_idx + 5]);
    bbox.set_ymax(gt_data[start_idx + 6]);
    (*all_gt_bboxes)[item_id].push_back(bbox);
  }
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::GetLocPredictions(const Dtype* loc_data,
      vector<LabelBBox>* loc_preds) {
  for (int i = 0; i < num_; ++i) {
    LabelBBox label_bbox;
    for (int p = 0; p < num_priors_; ++p) {
      int start_idx = p * loc_classes_ * 4;
      for (int c = 0; c < loc_classes_; ++c) {
        int label = share_location_ ? -1 : c;
        NormalizedBBox bbox;
        bbox.set_xmin(loc_data[start_idx + c * 4]);
        bbox.set_ymin(loc_data[start_idx + c * 4 + 1]);
        bbox.set_xmax(loc_data[start_idx + c * 4 + 2]);
        bbox.set_ymax(loc_data[start_idx + c * 4 + 3]);
        label_bbox[label].push_back(bbox);
      }
    }
    loc_data += num_priors_ * loc_classes_ * 4;
    loc_preds->push_back(label_bbox);
  }
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::GetPriorBBoxes(const Dtype* prior_data,
      vector<NormalizedBBox>* prior_bboxes,
      vector<vector<float> >* prior_variances) {
  for (int i = 0; i < num_priors_; ++i) {
    int start_idx = i * 4;
    NormalizedBBox bbox;
    bbox.set_xmin(prior_data[start_idx]);
    bbox.set_ymin(prior_data[start_idx + 1]);
    bbox.set_xmax(prior_data[start_idx + 2]);
    bbox.set_ymax(prior_data[start_idx + 3]);
    prior_bboxes->push_back(bbox);
  }

  for (int i = 0; i < num_priors_; ++i) {
    int start_idx = (num_priors_ + i) * 4;
    vector<float> var;
    for (int j = 0; j < 4; ++j) {
      var.push_back(prior_data[start_idx + j]);
    }
    prior_variances->push_back(var);
  }
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(false);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
  }
  const MultiBoxLossParameter& multibox_loss_param =
      this->layer_param_.multibox_loss_param();

  num_ = bottom[0]->num();
  num_priors_ = bottom[0]->height() / 4;
  // Get other parameters.
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  num_classes_ = multibox_loss_param.num_classes();
  CHECK_GE(num_classes_, 1) << "num_classes must not be less than 1.";
  share_location_ = multibox_loss_param.share_location();
  loc_classes_ = share_location_ ? 1 : num_classes_;
  match_type_ = multibox_loss_param.match_type();
  overlap_threshold_ = multibox_loss_param.overlap_threshold();
  use_prior_for_matching_ = multibox_loss_param.use_prior_for_matching();

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
  vector<int> conf_shape(2);
  conf_shape[0] = num_;
  conf_bottom_vec_.push_back(&conf_pred_);
  conf_bottom_vec_.push_back(&conf_gt_);
  conf_loss_.Reshape(loss_shape);
  conf_top_vec_.push_back(&conf_loss_);
  if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_conf");
    layer_param.set_type("SoftmaxWithLoss");
    layer_param.add_loss_weight(Dtype(1));
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(2);
    conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    conf_shape[1] = num_priors_;
    conf_gt_.Reshape(conf_shape);
    conf_shape.push_back(num_classes_);
    conf_pred_.Reshape(conf_shape);
    conf_loss_layer_->LayerSetUp(conf_bottom_vec_, conf_top_vec_);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  num_priors_ = bottom[0]->height() / 4;
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  CHECK_EQ(num_priors_ * loc_classes_ * 4, bottom[1]->height())
      << "Number of priors must match number of location predictions.";
  CHECK_EQ(num_priors_ * num_classes_, bottom[2]->height())
      << "Number of priors must match number of confidence predictions.";
  num_gt_ = bottom[3]->height();
  vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* prior_data = bottom[0]->cpu_data();
  const Dtype* loc_data = bottom[1]->cpu_data();
  const Dtype* gt_data = bottom[3]->cpu_data();

  // Retrieve all ground truth.
  map<int, vector<NormalizedBBox> > all_gt_bboxes;
  GetGroundTruth(gt_data, &all_gt_bboxes);

  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension.
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, &prior_bboxes, &prior_variances);

  // Retrieve all predictions.
  vector<LabelBBox> all_loc_preds;
  GetLocPredictions(loc_data, &all_loc_preds);

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
        if (match_indices[label][m] > -1) {
          num_matches_++;
        }
      }
    }
    all_match_indices_.push_back(match_indices);
    all_match_overlaps_.push_back(match_overlaps);
  }

  if (num_matches_ >= 1) {
    // Form data to pass on to loc_loss_layer_.
    vector<int> loc_shape(2);
    loc_shape[0] = 1;
    loc_shape[1] = num_matches_ * 4;
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
  vector<int> conf_shape(2);
  conf_shape[0] = num_;
  conf_shape[1] = num_priors_;
  conf_gt_.Reshape(conf_shape);
  conf_shape.push_back(num_classes_);
  conf_pred_.Reshape(conf_shape);
  CHECK_EQ(conf_pred_.count(), bottom[2]->count());
  // Directory copy the confidence prediction (but with different shape).
  caffe_copy(bottom[2]->count(), bottom[2]->cpu_data(),
             conf_pred_.mutable_cpu_data());
  Dtype* conf_gt_data = conf_gt_.mutable_cpu_data();
  if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    caffe_set(conf_gt_.count(), Dtype(0), conf_gt_data);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
  for (int i = 0; i < num_; ++i) {
    if (all_gt_bboxes.find(i) == all_gt_bboxes.end()) {
      conf_gt_data += conf_gt_.offset(1);
      continue;
    }
    const map<int, vector<int> >& match_indices = all_match_indices_[i];
    for (map<int, vector<int> >::const_iterator it = match_indices.begin();
         it != match_indices.end(); ++it) {
      const vector<int>& match_index = it->second;
      CHECK_EQ(match_index.size(), prior_bboxes.size());
      for (int j = 0; j < match_index.size(); ++j) {
        if (match_index[j] == -1) {
          continue;
        }
        const int gt_idx = match_index[j];
        conf_gt_data[j] = all_gt_bboxes[i][gt_idx].label();
      }
    }
    conf_gt_data += conf_gt_.offset(1);
  }
  conf_loss_layer_->Reshape(conf_bottom_vec_, conf_top_vec_);
  conf_loss_layer_->Forward(conf_bottom_vec_, conf_top_vec_);

  top[0]->mutable_cpu_data()[0] = 0;
  if (this->layer_param_.propagate_down(1)) {
    // TODO (weiliu89): Understand why it needs to divide 2.
    top[0]->mutable_cpu_data()[0] += loc_weight_ * loc_loss_.cpu_data()[0] / 2;
  }
  if (this->layer_param_.propagate_down(2)) {
    top[0]->mutable_cpu_data()[0] += conf_loss_.cpu_data()[0];
  }
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to prior inputs.";
  }
  if (propagate_down[3]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }

  // Back propagate on location prediction.
  if (propagate_down[1]) {
    vector<bool> loc_propagate_down;
    // Only back propagate on prediction, not ground truth.
    loc_propagate_down.push_back(true);
    loc_propagate_down.push_back(false);
    loc_loss_layer_->Backward(loc_top_vec_, loc_propagate_down,
                              loc_bottom_vec_);
    const Dtype* loc_pred_diff = loc_pred_.cpu_diff();
    Dtype* loc_bottom_diff = bottom[1]->mutable_cpu_diff();
    caffe_set(bottom[1]->count(), Dtype(0), loc_bottom_diff);
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
      loc_bottom_diff += bottom[1]->offset(1);
    }
  }

  // Back propagate on confidence prediction.
  if (propagate_down[2]) {
    vector<bool> conf_propagate_down;
    // Only back propagate on prediction, not ground truth.
    conf_propagate_down.push_back(true);
    conf_propagate_down.push_back(false);
    conf_loss_layer_->Backward(conf_top_vec_, conf_propagate_down,
                               conf_bottom_vec_);
    caffe_copy(conf_pred_.count(), conf_pred_.cpu_diff(),
               bottom[2]->mutable_cpu_diff());
  }
}


#ifdef CPU_ONLY
STUB_GPU(MultiBoxLossLayer);
#endif

INSTANTIATE_CLASS(MultiBoxLossLayer);

}  // namespace caffe
