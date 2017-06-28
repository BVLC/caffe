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

#include "caffe/util/bbox_util.hpp"

namespace caffe {

float BBoxSize(const NormalizedBBox& bbox, const bool normalized) {
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
                     const bool normalized) {
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


}  // namespace caffe
