#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#ifndef CAFFE_UTIL_BBOX_UTIL_H_
#define CAFFE_UTIL_BBOX_UTIL_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "caffe/caffe.hpp"

namespace caffe {

typedef map<int_tp, vector<NormalizedBBox> > LabelBBox;
typedef EmitConstraint_EmitType EmitType;
typedef PriorBoxParameter_CodeType CodeType;
typedef MultiBoxLossParameter_MatchType MatchType;

float BBoxSize(const NormalizedBBox& bbox, const bool normalized = true);
// Function used to sort NormalizedBBox, stored in STL container (e.g. vector),
// in ascend order based on the score value.
bool SortBBoxAscend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

// Function used to sort NormalizedBBox, stored in STL container (e.g. vector),
// in descend order based on the score value.
bool SortBBoxDescend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

// Function sued to sort pair<float, int>, stored in STL container (e.g. vector)
// in descend order based on the score (first) value.
bool SortScorePairAscend(const pair<float, int>& pair1,
                         const pair<float, int>& pair2);

// Function sued to sort pair<float, int>, stored in STL container (e.g. vector)
// in descend order based on the score (first) value.
bool SortScorePairDescend(const pair<float, int>& pair1,
                          const pair<float, int>& pair2);

// Generate unit bbox [0, 0, 1, 1]
NormalizedBBox UnitBBox();

// Compute the intersection between two bboxes.
void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                   NormalizedBBox* intersect_bbox);
float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                     const bool normalized);
void CumSum(const vector<pair<float, int_tp> >& pairs, vector<int_tp>* cumsum);
void ComputeAP(const vector<pair<float, int_tp> >& tp, int_tp num_pos,
               const vector<pair<float, int_tp> >& fp, string ap_version,
               vector<float>* prec, vector<float>* rec, float* ap);
                     
template <typename Dtype>
void setNormalizedBBox(NormalizedBBox& bbox, Dtype x, Dtype y, Dtype w, Dtype h)
{
  Dtype xmin = x - w/2.0;
  Dtype xmax = x + w/2.0;
  Dtype ymin = y - h/2.0;
  Dtype ymax = y + h/2.0;

  if (xmin < 0.0){
    xmin = 0.0;
  }
  if (xmax > 1.0){
    xmax = 1.0;
  }
  if (ymin < 0.0){
    ymin = 0.0;
  }
  if (ymax > 1.0){
    ymax = 1.0;
  }  
  bbox.set_xmin(xmin);
  bbox.set_ymin(ymin);
  bbox.set_xmax(xmax);
  bbox.set_ymax(ymax);
  float bbox_size = BBoxSize(bbox, true);
  bbox.set_size(bbox_size);
}

template <typename Dtype>
void GetDetectionResults(const Dtype* det_data, const int_tp num_det,
      map<int_tp, LabelBBox>* all_detections) {
  all_detections->clear();
  for (int_tp i = 0; i < num_det; ++i) {
    int_tp start_idx = i * 7;
    int_tp item_id = det_data[start_idx];
    if (item_id == -1) {
      continue;
    }
    int_tp label = det_data[start_idx + 1];
    NormalizedBBox bbox;
    Dtype x = det_data[start_idx + 3];
    Dtype y = det_data[start_idx + 4];
    Dtype w = det_data[start_idx + 5];
    Dtype h = det_data[start_idx + 6];

    setNormalizedBBox(bbox, x, y, w, h);
    bbox.set_score(det_data[start_idx + 2]); //confidence   
    (*all_detections)[item_id][label].push_back(bbox);
  }
}
template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int_tp num_gt,
      map<int_tp, LabelBBox >* all_gt_bboxes) {
  all_gt_bboxes->clear();
  int_tp cnt = 0;
  for (int_tp t = 0; t < 30; ++t){
    vector<Dtype> truth;
    int_tp label = gt_data[t * 5];
    Dtype x = gt_data[t * 5 + 1];
    Dtype y = gt_data[t * 5 + 2];
    Dtype w = gt_data[t * 5 + 3];
    Dtype h = gt_data[t * 5 + 4];

    if (!w) break;
    cnt++;
    int_tp item_id = 0;
    NormalizedBBox bbox;
    setNormalizedBBox(bbox, x, y, w, h);
    (*all_gt_bboxes)[item_id][label].push_back(bbox);
  }
}

template <typename Dtype>
Dtype Calc_rmse(const vector<Dtype>& box, const vector<Dtype>& truth) {
  return sqrt(pow(box[0]-truth[0], 2) +
              pow(box[1]-truth[1], 2) +
              pow(box[2]-truth[2], 2) +
              pow(box[3]-truth[3], 2));
}
template <typename Dtype>
Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
  Dtype l1 = x1 - w1/2;
  Dtype l2 = x2 - w2/2;
  Dtype left = l1 > l2 ? l1 : l2;
  Dtype r1 = x1 + w1/2;
  Dtype r2 = x2 + w2/2;
  Dtype right = r1 < r2 ? r1 : r2;
  return right - left;
}

template <typename Dtype>
Dtype Calc_iou(const vector<Dtype>& box, const vector<Dtype>& truth) {
  NormalizedBBox Bbox1, Bbox2;
  setNormalizedBBox(Bbox1, box[0], box[1], box[2], box[3]);
  setNormalizedBBox(Bbox2, truth[0], truth[1], truth[2], truth[3]);
  return JaccardOverlap(Bbox1, Bbox2, true);
}

template <typename Dtype>
void disp(Blob<Dtype>& swap)
{
  std::cout<<"#######################################"<<std::endl;
  for (int_tp b = 0; b < swap.num(); ++b)
    for (int_tp c = 0; c < swap.channels(); ++c)
      for (int_tp h = 0; h < swap.height(); ++h)
      {
  	std::cout<<"[";
        for (int_tp w = 0; w < swap.width(); ++w)
	{
	  std::cout<<swap.data_at(b,c,h,w)<<",";	
	}
	std::cout<<"]"<<std::endl;
      }
  return;
}


template <typename Dtype>
class PredictionResult{
  public:
    Dtype x;
    Dtype y;
    Dtype w;
    Dtype h;
    Dtype objScore;
    Dtype classScore;
    Dtype confidence;
    int_tp classType;
};
template <typename Dtype>
void class_index_and_score(Dtype* input, int_tp classes, PredictionResult<Dtype>& predict)
{
  Dtype sum = 0;
  Dtype large = input[0];
  int_tp classIndex = 0;
  for (int_tp i = 0; i < classes; ++i){
    if (input[i] > large)
      large = input[i];
  }
  for (int_tp i = 0; i < classes; ++i){
    Dtype e = exp(input[i] - large);
    sum += e;
    input[i] = e;
  }
  
  for (int_tp i = 0; i < classes; ++i){
    input[i] = input[i] / sum;   
  }
  large = input[0];
  classIndex = 0;

  for (int_tp i = 0; i < classes; ++i){
    if (input[i] > large){
      large = input[i];
      classIndex = i;
    }
  }  
  predict.classType = classIndex ;
  predict.classScore = large;
}
template <typename Dtype>
void get_region_box(Dtype* x, PredictionResult<Dtype>& predict, vector<Dtype> biases, int_tp n, int_tp index, int_tp i, int_tp j, int_tp w, int_tp h){
  predict.x = (i + sigmoid(x[index + 0])) / w;
  predict.y = (j + sigmoid(x[index + 1])) / h;
  predict.w = exp(x[index + 2]) * biases[2*n] / w;
  predict.h = exp(x[index + 3]) * biases[2*n+1] / h;
}
template <typename Dtype>
void ApplyNms(vector< PredictionResult<Dtype> >& boxes, vector<int_tp>& idxes, Dtype threshold) {
  map<int_tp, int_tp> idx_map;
  for (int_tp i = 0; i < boxes.size() - 1; ++i) {
    if (idx_map.find(i) != idx_map.end()) {
      continue;
    }
    for (int_tp j = i + 1; j < boxes.size(); ++j) {
      if (idx_map.find(j) != idx_map.end()) {
        continue;
      }
      NormalizedBBox Bbox1, Bbox2;
      setNormalizedBBox(Bbox1, boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h);
      setNormalizedBBox(Bbox2, boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h);

      float overlap = JaccardOverlap(Bbox1, Bbox2, true);

      if (overlap >= threshold) {
        idx_map[j] = 1;
      }
    }
  }
  for (int_tp i = 0; i < boxes.size(); ++i) {
    if (idx_map.find(i) == idx_map.end()) {
      idxes.push_back(i);
    }
  }
}

template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

// Explicit initialization.
template bool SortScorePairDescend(const pair<float, int_tp>& pair1,
                                   const pair<float, int_tp>& pair2);
template bool SortScorePairDescend(const pair<float, pair<int_tp, int_tp> >& pair1,
                                   const pair<float, pair<int_tp, int_tp> >& pair2);
              
typedef MultiBoxLossParameter_MatchType MatchType;
// Output the real bbox in the original image space.
void OutputBBox(const NormalizedBBox& bbox, const int height, const int width,
                const bool clip, NormalizedBBox* outbbox);
// Clip the NormalizedBBox such that the range for each corner is [0, 1].
void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clip_bbox);

// Scale the NormalizedBBox w.r.t. height and width.
void ScaleBBox(const NormalizedBBox& bbox, const int height, const int width,
               NormalizedBBox* scale_bbox);

// Locate bbox in the coordinate system that src_bbox sits.
void LocateBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
                NormalizedBBox* loc_bbox);

// Project bbox onto the coordinate system defined by src_bbox.
bool ProjectBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
                 NormalizedBBox* proj_bbox);

// Compute the coverage of bbox1 by bbox2.
float BBoxCoverage(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

// Encode a bbox according to a prior bbox.
void EncodeBBox(const NormalizedBBox& prior_bbox,
    const vector<float>& prior_variance, const CodeType code_type,
    const NormalizedBBox& bbox, NormalizedBBox* encode_bbox);

// Check if a bbox meet emit constraint w.r.t. src_bbox.
bool MeetEmitConstraint(const NormalizedBBox& src_bbox,
                        const NormalizedBBox& bbox,
                        const EmitConstraint& emit_constraint);

// Decode a bbox according to a prior bbox.
void DecodeBBox(const NormalizedBBox& prior_bbox,
    const vector<float>& prior_variance, const CodeType code_type,
    const NormalizedBBox& bbox, NormalizedBBox* decode_bbox);

// Decode a set of bboxes according to a set of prior bboxes.
void DecodeBBoxes(const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances, const CodeType code_type,
    const vector<NormalizedBBox>& bboxes,
    vector<NormalizedBBox>* decode_bboxes);

// Match prediction bboxes with ground truth bboxes.
void MatchBBox(const vector<NormalizedBBox>& gt,
    const vector<NormalizedBBox>& pred_bboxes, const int label,
    const MatchType match_type, const float overlap_threshold,
    vector<int>* match_indices, vector<float>* match_overlaps);

// Retrieve bounding box ground truth from gt_data.
//    gt_data: 1 x 1 x num_gt x 7 blob.
//    num_gt: the number of ground truth.
//    background_label_id: the label for background class which is used to do
//      santity check so that no ground truth contains it.
//    all_gt_bboxes: stores ground truth for each image. Label of each bbox is
//      stored in NormalizedBBox.
template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, vector<NormalizedBBox> >* all_gt_bboxes);
// Store ground truth bboxes of same label in a group.
template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, LabelBBox>* all_gt_bboxes);

// Get location predictions from loc_data.
//    loc_data: num x num_preds_per_class * num_loc_classes * 4 blob.
//    num: the number of images.
//    num_preds_per_class: number of predictions per class.
//    num_loc_classes: number of location classes. It is 1 if share_location is
//      true; and is equal to number of classes needed to predict otherwise.
//    share_location: if true, all classes share the same location prediction.
//    loc_preds: stores the location prediction, where each item contains
//      location prediction for an image.
template <typename Dtype>
void GetLocPredictions(const Dtype* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, vector<LabelBBox>* loc_preds);

// Get confidence predictions from conf_data.
//    conf_data: num x num_preds_per_class * num_classes blob.
//    num: the number of images.
//    num_preds_per_class: number of predictions per class.
//    num_classes: number of classes.
//    conf_preds: stores the confidence prediction, where each item contains
//      confidence prediction for an image.
template <typename Dtype>
void GetConfidenceScores(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      vector<map<int, vector<float> > >* conf_scores);

// Get max confidence scores for each prior from conf_data.
//    conf_data: num x num_preds_per_class * num_classes blob.
//    num: the number of images.
//    num_preds_per_class: number of predictions per class.
//    num_classes: number of classes.
//    prob: if true, compute the softmax probability.
//    all_max_scores: stores the max confidence per location for each image.
template <typename Dtype>
void GetMaxConfidenceScores(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes, const bool prob,
      vector<vector<float> >* all_max_scores);

// Get prior bounding boxes from prior_data.
//    prior_data: 1 x 2 x num_priors * 4 x 1 blob.
//    num_priors: number of priors.
//    prior_bboxes: stores all the prior bboxes in the format of NormalizedBBox.
//    prior_variances: stores all the variances needed by prior bboxes.
template <typename Dtype>
void GetPriorBBoxes(const Dtype* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes,
      vector<vector<float> >* prior_variances);

// Get detection results from det_data.
//    det_data: 1 x 1 x num_det x 7 blob.
//    num_det: the number of detections.
//    background_label_id: the label for background class which is used to do
//      santity check so that no detection contains it.
//    all_detections: stores detection results for each class from each image.
template <typename Dtype>
void GetDetectionResults(const Dtype* det_data, const int num_det,
      const int background_label_id,
      map<int, LabelBBox>* all_detections);

// Do non maximum suppression given bboxes and scores.
//    bboxes: a set of bounding boxes.
//    scores: a set of corresponding confidences.
//    threshold: the threshold used in non maximu suppression.
//    top_k: if not -1, keep at most top_k picked indices.
//    reuse_overlaps: if true, use and update overlaps; otherwise, always
//      compute overlap.
//    overlaps: a temp place to optionally store the overlaps between pairs of
//      bboxes if reuse_overlaps is true.
//    indices: the kept indices of bboxes after nms.
void ApplyNMS(const vector<NormalizedBBox>& bboxes, const vector<float>& scores,
      const float threshold, const int top_k, const bool reuse_overlaps,
      map<int, map<int, float> >* overlaps, vector<int>* indices);

// Compute cumsum of a set of pairs.
void CumSum(const vector<pair<float, int> >& pairs, vector<int>* cumsum);

// Compute average precision given true positive and false positive vectors.
//    tp: contains pairs of scores and true positive.
//    num_pos: number of positives.
//    fp: contains pairs of scores and false positive.
//    ap_version: different ways of computing Average Precision.
//      Check https://sanchom.wordpress.com/tag/average-precision/ for details.
//      11point: the 11-point interpolated average precision. Used in VOC2007.
//      MaxIntegral: maximally interpolated AP. Used in VOC2012/ILSVRC.
//      Integral: the natural integral of the precision-recall curve.
//    prec: stores the computed precisions.
//    rec: stores the computed recalls.
//    ap: the computed Average Precision.
void ComputeAP(const vector<pair<float, int> >& tp, const int num_pos,
               const vector<pair<float, int> >& fp, const string ap_version,
               vector<float>* prec, vector<float>* rec, float* ap);

}  // namespace caffe

#endif  // CAFFE_UTIL_BBOX_UTIL_H_
