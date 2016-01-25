#ifndef CAFFE_MULTIBOX_LOSS_LAYER_HPP_
#define CAFFE_MULTIBOX_LOSS_LAYER_HPP_

#include <map>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

typedef MultiBoxLossParameter_MatchType MatchType;
typedef MultiBoxLossParameter_LocLossType LocLossType;
typedef MultiBoxLossParameter_ConfLossType ConfLossType;
typedef map<int, vector<NormalizedBBox> > LabelBBox;

/**
 * @brief Perform MultiBox operations. Including the following:
 *
 *  - decode the predictions.
 *  - perform matching between priors/predictions and ground truth.
 *  - use matched boxes and confidences to compute loss.
 *
 */
template <typename Dtype>
class MultiBoxLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultiBoxLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiBoxLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // bottom[0] stores the prior bboxes.
  void GetPriorBBoxes(const Dtype* prior_data,
      vector<NormalizedBBox>* prior_boxes,
      vector<vector<float> >* prior_variances);
  // bottom[1] stores the location predictions.
  void GetLocPredictions(const Dtype* loc_data, vector<LabelBBox>* loc_preds);
  // Get all the inputs from bottom layers.
  // bottom[3] stores the ground truth bounding box labels.
  void GetGroundTruth(const Dtype* gt_data,
      map<int, vector<NormalizedBBox> >* all_gt_bboxes);

  // The internal localization loss layer.
  shared_ptr<Layer<Dtype> > loc_loss_layer_;
  LocLossType loc_loss_type_;
  float loc_weight_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> loc_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> loc_top_vec_;
  // blob which stores the matched location prediction.
  Blob<Dtype> loc_pred_;
  // blob which stores the corresponding matched ground truth.
  Blob<Dtype> loc_gt_;
  // localization loss.
  Blob<Dtype> loc_loss_;

  // The internal confidence loss layer.
  shared_ptr<Layer<Dtype> > conf_loss_layer_;
  ConfLossType conf_loss_type_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> conf_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> conf_top_vec_;
  // blob which stores the confidence prediction.
  Blob<Dtype> conf_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> conf_gt_;
  // confidence loss.
  Blob<Dtype> conf_loss_;

  int num_classes_;
  bool share_location_;
  MatchType match_type_;
  float overlap_threshold_;
  bool use_prior_for_matching_;
  int background_label_id_;

  int loc_classes_;
  int num_gt_;
  int num_;
  int num_priors_;

  vector<map<int, vector<int> > > all_match_indices_;
  vector<map<int, vector<float> > > all_match_overlaps_;
};

}  // namespace caffe

#endif  // CAFFE_MULTIBOX_LOSS_LAYER_HPP_
