#include <algorithm>
#include <csignal>
#include <ctime>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "boost/iterator/counting_iterator.hpp"

#include <map>
#include <vector>

#include "caffe/3rdparty/hungarian.h"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

float BBoxSize(const NormalizedBBox& bbox, const bool normalized = true) {
  if (bbox.xmax() < bbox.xmin() || bbox.ymax() < bbox.ymin()) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return 0;
  } else {
    if (bbox.has_size()) {
      return bbox.size();
    } else {
      float width = bbox.xmax() - bbox.xmin();
      float height = bbox.ymax() - bbox.ymin();
      if (normalized) {
        return width * height;
      } else {
        // If bbox is not within range [0, 1].
        return (width + 1) * (height + 1);
      }
    }
  }
}

void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                   NormalizedBBox* intersect_bbox) {
  if (bbox2.xmin() > bbox1.xmax() || bbox2.xmax() < bbox1.xmin() ||
      bbox2.ymin() > bbox1.ymax() || bbox2.ymax() < bbox1.ymin()) {
    // Return [0, 0, 0, 0] if there is no intersection.
    intersect_bbox->set_xmin(0);
    intersect_bbox->set_ymin(0);
    intersect_bbox->set_xmax(0);
    intersect_bbox->set_ymax(0);
  } else {
    intersect_bbox->set_xmin(std::max(bbox1.xmin(), bbox2.xmin()));
    intersect_bbox->set_ymin(std::max(bbox1.ymin(), bbox2.ymin()));
    intersect_bbox->set_xmax(std::min(bbox1.xmax(), bbox2.xmax()));
    intersect_bbox->set_ymax(std::min(bbox1.ymax(), bbox2.ymax()));
  }
}

float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                     const bool normalized = true) {
  NormalizedBBox intersect_bbox;
  IntersectBBox(bbox1, bbox2, &intersect_bbox);
  float intersect_width, intersect_height;
  if (normalized) {
    intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin();
    intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin();
  } else {
    intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin() + 1;
    intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin() + 1;
  }
  if (intersect_width > 0 && intersect_height > 0) {
    float intersect_size = intersect_width * intersect_height;
    float bbox1_size = BBoxSize(bbox1, normalized);
    float bbox2_size = BBoxSize(bbox2, normalized);
    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
  } else {
    return 0.;
  }
}

bool SortBBoxAscend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
  return bbox1.score() < bbox2.score();
}

bool SortBBoxDescend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
  return bbox1.score() > bbox2.score();
}

void CumSum(const vector<pair<float, int_tp> >& pairs, vector<int_tp>* cumsum) {
  // Sort the pairs based on first item of the pair.
  vector<pair<float, int_tp> > sort_pairs = pairs;
  std::stable_sort(sort_pairs.begin(), sort_pairs.end(),
                   SortScorePairDescend<int_tp>);

  cumsum->clear();
  for (int_tp i = 0; i < sort_pairs.size(); ++i) {
    if (i == 0) {
      cumsum->push_back(sort_pairs[i].second);
    } else {
      cumsum->push_back(cumsum->back() + sort_pairs[i].second);
    }
  }
}
void ComputeAP(const vector<pair<float, int_tp> >& tp, int_tp num_pos,
               const vector<pair<float, int_tp> >& fp, string ap_version,
               vector<float>* prec, vector<float>* rec, float* ap) {
  const float eps = 1e-6;
  CHECK_EQ(tp.size(), fp.size()) << "tp must have same size as fp.";
  const int_tp num = tp.size();
  // Make sure that tp and fp have complement value.
  for (int_tp i = 0; i < num; ++i) {
    CHECK_LE(fabs(tp[i].first - fp[i].first), eps);
    CHECK_GE(tp[i].second, 0);
    CHECK_GE(fp[i].second, 0);
  }
  prec->clear();
  rec->clear();
  *ap = 0;
  if (tp.size() == 0 || num_pos == 0) {
    return;
  }

  // Compute cumsum of tp.
  vector<int_tp> tp_cumsum;
  CumSum(tp, &tp_cumsum);
  CHECK_EQ(tp_cumsum.size(), num);

  // Compute cumsum of fp.
  vector<int_tp> fp_cumsum;
  CumSum(fp, &fp_cumsum);
  CHECK_EQ(fp_cumsum.size(), num);

  // Compute precision.
  for (int_tp i = 0; i < num; ++i) {
    prec->push_back(static_cast<float>(tp_cumsum[i]) /
                    (tp_cumsum[i] + fp_cumsum[i]));
  }

  // Compute recall.
  for (int_tp i = 0; i < num; ++i) {
    CHECK_LE(tp_cumsum[i], num_pos);
    rec->push_back(static_cast<float>(tp_cumsum[i]) / num_pos);
  }

  // for (int_tp i = 0; i < num; ++i) {
  //   std::cout << (*prec)[i] << std::endl;
  //   std::cout << (*rec)[i] << std::endl;
  // }

  if (ap_version == "11point") {
    // VOC2007 style for computing AP.
    vector<float> max_precs(11, 0.);
    int_tp start_idx = num - 1;
    for (int_tp j = 10; j >= 0; --j) {
      for (int_tp i = start_idx; i >= 0 ; --i) {
        if ((*rec)[i] < j / 10.) {
          start_idx = i;
          if (j > 0) {
            max_precs[j-1] = max_precs[j];
          }
          break;
        } else {
          if (max_precs[j] < (*prec)[i]) {
            max_precs[j] = (*prec)[i];
          }
        }
      }
    }
    for (int_tp j = 10; j >= 0; --j) {
      *ap += max_precs[j] / 11;
    }
  } else if (ap_version == "MaxIntegral") {
    // VOC2012 or ILSVRC style for computing AP.
    float cur_rec = rec->back();
    float cur_prec = prec->back();
    for (int_tp i = num - 2; i >= 0; --i) {
      cur_prec = std::max<float>((*prec)[i], cur_prec);
      if (fabs(cur_rec - (*rec)[i]) > eps) {
        *ap += cur_prec * fabs(cur_rec - (*rec)[i]);
      }
      cur_rec = (*rec)[i];
    }
    *ap += cur_rec * cur_prec;
  } else if (ap_version == "Integral") {
    // Natural integral.
    float prev_rec = 0.;
    for (int_tp i = 0; i < num; ++i) {
      if (fabs((*rec)[i] - prev_rec) > eps) {
        *ap += (*prec)[i] * fabs((*rec)[i] - prev_rec);
      }
      prev_rec = (*rec)[i];
    }
  } else {
    LOG(FATAL) << "Unknown ap_version: " << ap_version;
  }
}

void EncodeBBox(
    const NormalizedBBox& prior_bbox, const vector<float>& prior_variance,
    const NormalizedBBox& bbox, NormalizedBBox* encode_bbox) {
  CHECK_EQ(prior_variance.size(), 4);
  for (int i = 0; i < prior_variance.size(); ++i) {
    CHECK_GT(prior_variance[i], 0);
  }
  encode_bbox->set_xmin((bbox.xmin() - prior_bbox.xmin()) / prior_variance[0]);
  encode_bbox->set_ymin((bbox.ymin() - prior_bbox.ymin()) / prior_variance[1]);
  encode_bbox->set_xmax((bbox.xmax() - prior_bbox.xmax()) / prior_variance[2]);
  encode_bbox->set_ymax((bbox.ymax() - prior_bbox.ymax()) / prior_variance[3]);
}

void DecodeBBox(
    const NormalizedBBox& prior_bbox, const vector<float>& prior_variance,
    const NormalizedBBox& bbox, NormalizedBBox* decode_bbox) {
  decode_bbox->set_xmin(prior_bbox.xmin() + prior_variance[0] * bbox.xmin());
  decode_bbox->set_ymin(prior_bbox.ymin() + prior_variance[1] * bbox.ymin());
  decode_bbox->set_xmax(prior_bbox.xmax() + prior_variance[2] * bbox.xmax());
  decode_bbox->set_ymax(prior_bbox.ymax() + prior_variance[3] * bbox.ymax());
}

void DecodeBBoxes(
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const vector<NormalizedBBox>& bboxes,
    vector<NormalizedBBox>* decode_bboxes) {
  CHECK_EQ(prior_bboxes.size(), prior_variances.size());
  CHECK_EQ(prior_bboxes.size(), bboxes.size());
  int num_bboxes = prior_bboxes.size();
  if (num_bboxes >= 1) {
    CHECK_EQ(prior_variances[0].size(), 4);
  }
  decode_bboxes->clear();
  for (int i = 0; i < num_bboxes; ++i) {
    NormalizedBBox decode_bbox;
    DecodeBBox(prior_bboxes[i], prior_variances[i], bboxes[i],
               &decode_bbox);
    decode_bboxes->push_back(decode_bbox);
  }
}

void MatchBBox(const vector<NormalizedBBox>& gt_bboxes,
    const vector<NormalizedBBox>& pred_bboxes, const int label,
    const MatchType match_type, const float overlap_threshold,
    vector<int>* match_indices, vector<float>* match_overlaps) {
  int num_pred = pred_bboxes.size();
  match_indices->clear();
  match_indices->resize(num_pred, -1);
  match_overlaps->clear();
  match_overlaps->resize(num_pred, 0.);

  int num_gt = 0;
  vector<int> gt_indices;
  if (label == -1) {
    // label -1 means comparing against all ground truth.
    num_gt = gt_bboxes.size();
    for (int i = 0; i < num_gt; ++i) {
      gt_indices.push_back(i);
    }
  } else {
    // Count number of ground truth boxes which has the desired label.
    for (int i = 0; i < gt_bboxes.size(); ++i) {
      if (gt_bboxes[i].label() == label) {
        num_gt++;
        gt_indices.push_back(i);
      }
    }
  }
  if (num_gt == 0) {
    return;
  }

  // Store the positive overlap between predictions and ground truth.
  map<int, map<int, float> > overlaps;
  for (int i = 0; i < num_pred; ++i) {
    for (int j = 0; j < num_gt; ++j) {
      float overlap = JaccardOverlap(pred_bboxes[i], gt_bboxes[gt_indices[j]]);
      if (overlap > 1e-6) {
        overlaps[i][j] = overlap;
      }
    }
  }

  // Bipartite matching.
  vector<int> gt_pool;
  for (int i = 0; i < num_gt; ++i) {
    gt_pool.push_back(i);
  }
  while (gt_pool.size() > 0) {
    // Find the most overlapped gt and cooresponding predictions.
    int max_idx = -1;
    int max_gt_idx = -1;
    float max_overlap = -1;
    for (map<int, map<int, float> >::iterator it = overlaps.begin();
         it != overlaps.end(); ++it) {
      int i = it->first;
      if ((*match_indices)[i] != -1) {
        // The prediction already have matched ground truth.
        continue;
      }
      for (int p = 0; p < gt_pool.size(); ++p) {
        int j = gt_pool[p];
        if (it->second.find(j) == it->second.end()) {
          // No overlap between the i-th prediction and j-th ground truth.
          continue;
        }
        // Find the maximum overlapped pair.
        if (it->second[j] > max_overlap) {
          // If the prediction has not been matched to any ground truth,
          // and the overlap is larger than maximum overlap, update.
          max_idx = i;
          max_gt_idx = j;
          max_overlap = it->second[j];
        }
      }
    }
    if (max_idx == -1) {
      // Cannot find good match.
      break;
    } else {
      CHECK_EQ((*match_indices)[max_idx], -1);
      (*match_indices)[max_idx] = gt_indices[max_gt_idx];
      (*match_overlaps)[max_idx] = max_overlap;
      // Erase the ground truth.
      gt_pool.erase(std::find(gt_pool.begin(), gt_pool.end(), max_gt_idx));
    }
  }

  switch (match_type) {
    case MultiBoxLossParameter_MatchType_BIPARTITE:
      // Already done.
      break;
    case MultiBoxLossParameter_MatchType_PER_PREDICTION:
      // Get most overlaped for the rest prediction bboxes.
      for (map<int, map<int, float> >::iterator it = overlaps.begin();
           it != overlaps.end(); ++it) {
        int i = it->first;
        if ((*match_indices)[i] != -1) {
          // The prediction already have matched ground truth.
          continue;
        }
        int max_gt_idx = -1;
        float max_overlap = -1;
        for (int j = 0; j < num_gt; ++j) {
          if (it->second.find(j) == it->second.end()) {
            // No overlap between the i-th prediction and j-th ground truth.
            continue;
          }
          // Find the maximum overlapped pair.
          float overlap = it->second[j];
          if (overlap >= overlap_threshold && overlap > max_overlap) {
            // If the prediction has not been matched to any ground truth,
            // and the overlap is larger than maximum overlap, update.
            max_gt_idx = j;
            max_overlap = overlap;
          }
        }
        if (max_gt_idx != -1) {
          // Found a matched ground truth.
          CHECK_EQ((*match_indices)[i], -1);
          (*match_indices)[i] = gt_indices[max_gt_idx];
          (*match_overlaps)[i] = max_overlap;
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown matching type.";
      break;
  }

  return;
}

template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int num_gt,
      const int background_label_id,
      map<int, vector<NormalizedBBox> >* all_gt_bboxes) {
  all_gt_bboxes->clear();
  for (int i = 0; i < num_gt; ++i) {
    int start_idx = i * 7;
    int item_id = gt_data[start_idx];
    if (item_id == -1) {
      break;
    }
    NormalizedBBox bbox;
    bbox.set_label(gt_data[start_idx + 1]);
    CHECK_NE(background_label_id, bbox.label())
        << "Found background label in the dataset.";
    bbox.set_xmin(gt_data[start_idx + 3]);
    bbox.set_ymin(gt_data[start_idx + 4]);
    bbox.set_xmax(gt_data[start_idx + 5]);
    bbox.set_ymax(gt_data[start_idx + 6]);
    (*all_gt_bboxes)[item_id].push_back(bbox);
  }
}

// Explicit initialization.
template void GetGroundTruth(const float* gt_data, const int num_gt,
      const int background_label_id,
      map<int, vector<NormalizedBBox> >* all_gt_bboxes);
template void GetGroundTruth(const double* gt_data, const int num_gt,
      const int background_label_id,
      map<int, vector<NormalizedBBox> >* all_gt_bboxes);

template <typename Dtype>
void GetLocPredictions(const Dtype* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, vector<LabelBBox>* loc_preds) {
  loc_preds->clear();
  for (int i = 0; i < num; ++i) {
    LabelBBox label_bbox;
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_loc_classes * 4;
      for (int c = 0; c < num_loc_classes; ++c) {
        int label = share_location ? -1 : c;
        NormalizedBBox bbox;
        bbox.set_xmin(loc_data[start_idx + c * 4]);
        bbox.set_ymin(loc_data[start_idx + c * 4 + 1]);
        bbox.set_xmax(loc_data[start_idx + c * 4 + 2]);
        bbox.set_ymax(loc_data[start_idx + c * 4 + 3]);
        label_bbox[label].push_back(bbox);
      }
    }
    loc_data += num_preds_per_class * num_loc_classes * 4;
    loc_preds->push_back(label_bbox);
  }
}

// Explicit initialization.
template void GetLocPredictions(const float* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, vector<LabelBBox>* loc_preds);
template void GetLocPredictions(const double* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, vector<LabelBBox>* loc_preds);

template <typename Dtype>
void GetPriorBBoxes(const Dtype* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes,
      vector<vector<float> >* prior_variances) {
  prior_bboxes->clear();
  prior_variances->clear();
  for (int i = 0; i < num_priors; ++i) {
    int start_idx = i * 4;
    NormalizedBBox bbox;
    bbox.set_xmin(prior_data[start_idx]);
    bbox.set_ymin(prior_data[start_idx + 1]);
    bbox.set_xmax(prior_data[start_idx + 2]);
    bbox.set_ymax(prior_data[start_idx + 3]);
    prior_bboxes->push_back(bbox);
  }

  for (int i = 0; i < num_priors; ++i) {
    int start_idx = (num_priors + i) * 4;
    vector<float> var;
    for (int j = 0; j < 4; ++j) {
      var.push_back(prior_data[start_idx + j]);
    }
    prior_variances->push_back(var);
  }
}

// Explicit initialization.
template void GetPriorBBoxes(const float* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes,
      vector<vector<float> >* prior_variances);
template void GetPriorBBoxes(const double* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes,
      vector<vector<float> >* prior_variances);

}  // namespace caffe
