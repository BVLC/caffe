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
  int num_pos = overlaps.size();

  // Create costs matrix to be used by libhungarian. Since libhungarian only
  // accept integer cost matrix, we scale overlap appropriately.
  float scale = 1e5;
  int** costs = new int*[num_gt];
  for (int i = 0; i < num_gt; ++i) {
    costs[i] = new int[num_pos];
    int j = 0;
    for (map<int, map<int, float> >::iterator it = overlaps.begin();
         it != overlaps.end(); ++it, ++j) {
      if (it->second.find(i) == it->second.end()) {
        costs[i][j] = 0;
      } else {
        costs[i][j] = static_cast<int>(it->second[i] * scale);
      }
    }
  }

  // Use hungarian algorithm to solve bipartite matching.
  hungarian_problem_t p;
  hungarian_init(&p, costs, num_gt, num_pos, HUNGARIAN_MODE_MAXIMIZE_UTIL);
  hungarian_solve(&p);

  // Output match results. Since currently both BIPARTITE and PER_PREDICTION
  // matching method need to perform BIPARTITE matching, we put it outside.
  for (int i = 0; i < num_gt; ++i) {
    int j = 0;
    for (map<int, map<int, float> >::iterator it = overlaps.begin();
         it != overlaps.end(); ++it, ++j) {
      int pred_idx = it->first;
      if (p.assignment[i][j] == HUNGARIAN_ASSIGNED) {
        CHECK_EQ((*match_indices)[pred_idx], -1) << "Found multiple matches";
        if (it->second.find(i) == it->second.end()) {
          continue;
        }
        (*match_indices)[pred_idx] = gt_indices[i];
        (*match_overlaps)[pred_idx] = it->second[i];
      }
    }
  }
  switch (match_type) {
    case MultiBoxLossParameter_MatchType_BIPARTITE:
      // Already done.
      break;
    case MultiBoxLossParameter_MatchType_PER_PREDICTION:
      // Get most overlaped for the rest prediction bboxes.
      for (int i = 0; i < num_gt; ++i) {
        for (map<int, map<int, float> >::iterator it = overlaps.begin();
             it != overlaps.end(); ++it) {
          int pred_idx = it->first;
          if ((*match_indices)[pred_idx] > -1) {
            // Already found a match during Bipartite matching step.
            continue;
          }
          if (it->second.find(i) == it->second.end()) {
            if (overlap_threshold == 0) {
              (*match_indices)[pred_idx] = gt_indices[i];
              (*match_overlaps)[pred_idx] = 0;
            }
          } else {
            if (it->second[i] >= overlap_threshold &&
                it->second[i] >= (*match_overlaps)[pred_idx]) {
              (*match_indices)[pred_idx] = gt_indices[i];
              (*match_overlaps)[pred_idx] = it->second[i];
            }
          }
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown matching type.";
      break;
  }

  // free space
  hungarian_free(&p);
  for (int i = 0; i < num_gt; ++i) {
    free(costs[i]);
  }
  free(costs);

  return;
}

}  // namespace caffe
