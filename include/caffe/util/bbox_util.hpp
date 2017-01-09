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

typedef EmitConstraint_EmitType EmitType;
typedef PriorBoxParameter_CodeType CodeType;
typedef MultiBoxLossParameter_MatchType MatchType;
typedef MultiBoxLossParameter_LocLossType LocLossType;
typedef MultiBoxLossParameter_ConfLossType ConfLossType;

typedef map<int, vector<NormalizedBBox> > LabelBBox;

// Function used to sort NormalizedBBox, stored in STL container (e.g. vector),
// in ascend order based on the score value.
bool SortBBoxAscend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

// Function used to sort NormalizedBBox, stored in STL container (e.g. vector),
// in descend order based on the score value.
bool SortBBoxDescend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

// Function sued to sort pair<float, T>, stored in STL container (e.g. vector)
// in descend order based on the score (first) value.
template <typename T>
bool SortScorePairAscend(const pair<float, T>& pair1,
                         const pair<float, T>& pair2);

// Function sued to sort pair<float, T>, stored in STL container (e.g. vector)
// in descend order based on the score (first) value.
template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2);

// Generate unit bbox [0, 0, 1, 1]
NormalizedBBox UnitBBox();

// Compute the intersection between two bboxes.
void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                   NormalizedBBox* intersect_bbox);

// Compute bbox size.
float BBoxSize(const NormalizedBBox& bbox, const bool normalized = true);

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

// Compute the jaccard (intersection over union IoU) overlap between two bboxes.
float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                     const bool normalized = true);

// Compute the coverage of bbox1 by bbox2.
float BBoxCoverage(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

// Encode a bbox according to a prior bbox.
void EncodeBBox(const NormalizedBBox& prior_bbox,
    const vector<float>& prior_variance, const CodeType code_type,
    const bool encode_variance_in_target, const NormalizedBBox& bbox,
    NormalizedBBox* encode_bbox);

// Check if a bbox meet emit constraint w.r.t. src_bbox.
bool MeetEmitConstraint(const NormalizedBBox& src_bbox,
    const NormalizedBBox& bbox, const EmitConstraint& emit_constraint);

// Decode a bbox according to a prior bbox.
void DecodeBBox(const NormalizedBBox& prior_bbox,
    const vector<float>& prior_variance, const CodeType code_type,
    const bool variance_encoded_in_target, const NormalizedBBox& bbox,
    NormalizedBBox* decode_bbox);

// Decode a set of bboxes according to a set of prior bboxes.
void DecodeBBoxes(const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const CodeType code_type, const bool variance_encoded_in_target,
    const vector<NormalizedBBox>& bboxes,
    vector<NormalizedBBox>* decode_bboxes);

// Decode all bboxes in a batch.
void DecodeBBoxesAll(const vector<LabelBBox>& all_loc_pred,
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const int num, const bool share_location,
    const int num_loc_classes, const int background_label_id,
    const CodeType code_type, const bool variance_encoded_in_target,
    vector<LabelBBox>* all_decode_bboxes);

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

// Get confidence predictions from conf_data.
//    conf_data: num x num_preds_per_class * num_classes blob.
//    num: the number of images.
//    num_preds_per_class: number of predictions per class.
//    num_classes: number of classes.
//    class_major: if true, data layout is
//      num x num_classes x num_preds_per_class; otherwise, data layerout is
//      num x num_preds_per_class * num_classes.
//    conf_preds: stores the confidence prediction, where each item contains
//      confidence prediction for an image.
template <typename Dtype>
void GetConfidenceScores(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const bool class_major, vector<map<int, vector<float> > >* conf_scores);

// Get max confidence scores for each prior from conf_data.
//    conf_data: num x num_preds_per_class * num_classes blob.
//    num: the number of images.
//    num_preds_per_class: number of predictions per class.
//    num_classes: number of classes.
//    background_label_id: it is used to skip selecting max scores from
//      background class.
//    loss_type: compute the probability according to the loss type.
//    all_max_scores: stores the max confidence per location for each image.
template <typename Dtype>
void GetMaxConfidenceScores(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
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

// Get top_k scores with corresponding indices.
//    scores: a set of scores.
//    top_k: if -1, keep all; otherwise, keep at most top_k.
//    score_index_vec: store the sorted (score, index) pair.
void GetTopKScoreIndex(const vector<float>& scores, const int top_k,
                         vector<pair<float, int> >* score_index_vec);

// Get max scores with corresponding indices.
//    scores: a set of scores.
//    threshold: only consider scores higher than the threshold.
//    top_k: if -1, keep all; otherwise, keep at most top_k.
//    score_index_vec: store the sorted (score, index) pair.
void GetMaxScoreIndex(const vector<float>& scores, const float threshold,
      const int top_k, vector<pair<float, int> >* score_index_vec);

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

void ApplyNMS(const bool* overlapped, const int num, vector<int>* indices);

// Do non maximum suppression given bboxes and scores.
// Inspired by Piotr Dollar's NMS implementation in EdgeBox.
// https://goo.gl/jV3JYS
//    bboxes: a set of bounding boxes.
//    scores: a set of corresponding confidences.
//    score_threshold: a threshold used to filter detection results.
//    nms_threshold: a threshold used in non maximum suppression.
//    top_k: if not -1, keep at most top_k picked indices.
//    indices: the kept indices of bboxes after nms.
void ApplyNMSFast(const vector<NormalizedBBox>& bboxes,
      const vector<float>& scores, const float score_threshold,
      const float nms_threshold, const int top_k, vector<int>* indices);

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

#ifndef CPU_ONLY  // GPU
template <typename Dtype>
__host__ __device__ Dtype BBoxSizeGPU(const Dtype* bbox,
                                      const bool normalized = true);

template <typename Dtype>
__host__ __device__ Dtype JaccardOverlapGPU(const Dtype* bbox1,
                                            const Dtype* bbox2);

template <typename Dtype>
void DecodeBBoxesGPU(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          Dtype* bbox_data);

template <typename Dtype>
void PermuteDataGPU(const int nthreads,
          const Dtype* data, const int num_classes, const int num_data,
          const int num_dim, Dtype* new_data);

template <typename Dtype>
void ComputeOverlappedGPU(const int nthreads,
          const Dtype* bbox_data, const int num_bboxes, const int num_classes,
          const Dtype overlap_threshold, bool* overlapped_data);

template <typename Dtype>
void ComputeOverlappedByIdxGPU(const int nthreads,
          const Dtype* bbox_data, const Dtype overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data);

template <typename Dtype>
void ApplyNMSGPU(const Dtype* bbox_data, const Dtype* conf_data,
          const int num_bboxes, const float confidence_threshold,
          const int top_k, const float nms_threshold, vector<int>* indices);

template <typename Dtype>
void GetDetectionsGPU(const Dtype* bbox_data, const Dtype* conf_data,
          const int image_id, const int label, const vector<int>& indices,
          const bool clip_bbox, Blob<Dtype>* detection_blob);
#endif  // !CPU_ONLY

#ifdef USE_OPENCV
vector<cv::Scalar> GetColors(const int n);

template <typename Dtype>
void VisualizeBBox(const vector<cv::Mat>& images, const Blob<Dtype>* detections,
                   const float threshold, const vector<cv::Scalar>& colors,
                   const map<int, string>& label_to_display_name);
#endif  // USE_OPENCV

}  // namespace caffe

#endif  // CAFFE_UTIL_BBOX_UTIL_H_
