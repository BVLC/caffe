#include <map>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/util/bbox_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

static const float eps = 1e-6;

void FillBBoxes(vector<NormalizedBBox>* gt_bboxes,
                vector<NormalizedBBox>* pred_bboxes) {
  gt_bboxes->clear();
  pred_bboxes->clear();
  NormalizedBBox bbox;

  // Fill in ground truth bboxes.
  bbox.set_label(1);
  bbox.set_xmin(0.1);
  bbox.set_ymin(0.1);
  bbox.set_xmax(0.3);
  bbox.set_ymax(0.3);
  gt_bboxes->push_back(bbox);

  bbox.set_label(2);
  bbox.set_xmin(0.3);
  bbox.set_ymin(0.3);
  bbox.set_xmax(0.6);
  bbox.set_ymax(0.5);
  gt_bboxes->push_back(bbox);

  // Fill in prediction bboxes.
  bbox.set_xmin(0.1);
  bbox.set_ymin(0);
  bbox.set_xmax(0.4);
  bbox.set_ymax(0.3);
  pred_bboxes->push_back(bbox);

  bbox.set_xmin(0);
  bbox.set_ymin(0.1);
  bbox.set_xmax(0.2);
  bbox.set_ymax(0.3);
  pred_bboxes->push_back(bbox);

  bbox.set_xmin(0.2);
  bbox.set_ymin(0.1);
  bbox.set_xmax(0.4);
  bbox.set_ymax(0.4);
  pred_bboxes->push_back(bbox);

  bbox.set_xmin(0.4);
  bbox.set_ymin(0.3);
  bbox.set_xmax(0.7);
  bbox.set_ymax(0.5);
  pred_bboxes->push_back(bbox);

  bbox.set_xmin(0.5);
  bbox.set_ymin(0.4);
  bbox.set_xmax(0.7);
  bbox.set_ymax(0.7);
  pred_bboxes->push_back(bbox);

  bbox.set_xmin(0.7);
  bbox.set_ymin(0.7);
  bbox.set_xmax(0.8);
  bbox.set_ymax(0.8);
  pred_bboxes->push_back(bbox);
}

class BBoxUtilTest : public ::testing::Test {};

TEST_F(BBoxUtilTest, TestIntersectBBox) {
  NormalizedBBox bbox_ref;
  bbox_ref.set_xmin(0.2);
  bbox_ref.set_ymin(0.3);
  bbox_ref.set_xmax(0.3);
  bbox_ref.set_ymax(0.5);

  NormalizedBBox bbox_test;
  NormalizedBBox bbox_intersect;

  // Partially overlapped.
  bbox_test.set_xmin(0.1);
  bbox_test.set_ymin(0.1);
  bbox_test.set_xmax(0.3);
  bbox_test.set_ymax(0.4);
  IntersectBBox(bbox_ref, bbox_test, &bbox_intersect);
  EXPECT_NEAR(bbox_intersect.xmin(), 0.2, eps);
  EXPECT_NEAR(bbox_intersect.ymin(), 0.3, eps);
  EXPECT_NEAR(bbox_intersect.xmax(), 0.3, eps);
  EXPECT_NEAR(bbox_intersect.ymax(), 0.4, eps);

  // Fully contain.
  bbox_test.set_xmin(0.1);
  bbox_test.set_ymin(0.1);
  bbox_test.set_xmax(0.4);
  bbox_test.set_ymax(0.6);
  IntersectBBox(bbox_ref, bbox_test, &bbox_intersect);
  EXPECT_NEAR(bbox_intersect.xmin(), 0.2, eps);
  EXPECT_NEAR(bbox_intersect.ymin(), 0.3, eps);
  EXPECT_NEAR(bbox_intersect.xmax(), 0.3, eps);
  EXPECT_NEAR(bbox_intersect.ymax(), 0.5, eps);

  // Outside.
  bbox_test.set_xmin(0);
  bbox_test.set_ymin(0);
  bbox_test.set_xmax(0.1);
  bbox_test.set_ymax(0.1);
  IntersectBBox(bbox_ref, bbox_test, &bbox_intersect);
  EXPECT_NEAR(bbox_intersect.xmin(), 0, eps);
  EXPECT_NEAR(bbox_intersect.ymin(), 0, eps);
  EXPECT_NEAR(bbox_intersect.xmax(), 0, eps);
  EXPECT_NEAR(bbox_intersect.ymax(), 0, eps);
}

TEST_F(BBoxUtilTest, TestBBoxSize) {
  NormalizedBBox bbox;
  float size;

  // Valid box.
  bbox.set_xmin(0.2);
  bbox.set_ymin(0.3);
  bbox.set_xmax(0.3);
  bbox.set_ymax(0.5);
  size = BBoxSize(bbox);
  EXPECT_NEAR(size, 0.02, eps);

  // A line.
  bbox.set_xmin(0.2);
  bbox.set_ymin(0.3);
  bbox.set_xmax(0.2);
  bbox.set_ymax(0.5);
  size = BBoxSize(bbox);
  EXPECT_NEAR(size, 0., eps);

  // Invalid box.
  bbox.set_xmin(0.2);
  bbox.set_ymin(0.3);
  bbox.set_xmax(0.1);
  bbox.set_ymax(0.5);
  size = BBoxSize(bbox);
  EXPECT_NEAR(size, 0., eps);
}

TEST_F(BBoxUtilTest, TestJaccardOverlap) {
  NormalizedBBox bbox1;
  bbox1.set_xmin(0.2);
  bbox1.set_ymin(0.3);
  bbox1.set_xmax(0.3);
  bbox1.set_ymax(0.5);

  NormalizedBBox bbox2;
  float overlap;

  // Partially overlapped.
  bbox2.set_xmin(0.1);
  bbox2.set_ymin(0.1);
  bbox2.set_xmax(0.3);
  bbox2.set_ymax(0.4);
  overlap = JaccardOverlap(bbox1, bbox2);
  EXPECT_NEAR(overlap, 1./7, eps);

  // Fully contain.
  bbox2.set_xmin(0.1);
  bbox2.set_ymin(0.1);
  bbox2.set_xmax(0.4);
  bbox2.set_ymax(0.6);
  overlap = JaccardOverlap(bbox1, bbox2);
  EXPECT_NEAR(overlap, 2./15, eps);

  // Outside.
  bbox2.set_xmin(0);
  bbox2.set_ymin(0);
  bbox2.set_xmax(0.1);
  bbox2.set_ymax(0.1);
  overlap = JaccardOverlap(bbox1, bbox2);
  EXPECT_NEAR(overlap, 0., eps);
}

TEST_F(BBoxUtilTest, TestEncodeBBox) {
  NormalizedBBox prior_bbox;
  prior_bbox.set_xmin(0.1);
  prior_bbox.set_ymin(0.1);
  prior_bbox.set_xmax(0.3);
  prior_bbox.set_ymax(0.3);
  vector<float> prior_variance(4, 0.1);

  NormalizedBBox bbox;
  bbox.set_xmin(0);
  bbox.set_ymin(0.2);
  bbox.set_xmax(0.4);
  bbox.set_ymax(0.5);

  NormalizedBBox encode_bbox;
  EncodeBBox(prior_bbox, prior_variance, bbox, &encode_bbox);

  EXPECT_NEAR(encode_bbox.xmin(), -1, eps);
  EXPECT_NEAR(encode_bbox.ymin(), 1, eps);
  EXPECT_NEAR(encode_bbox.xmax(), 1, eps);
  EXPECT_NEAR(encode_bbox.ymax(), 2, eps);
}

TEST_F(BBoxUtilTest, TestDecodeBBox) {
  NormalizedBBox prior_bbox;
  prior_bbox.set_xmin(0.1);
  prior_bbox.set_ymin(0.1);
  prior_bbox.set_xmax(0.3);
  prior_bbox.set_ymax(0.3);
  vector<float> prior_variance(4, 0.1);

  NormalizedBBox bbox;
  bbox.set_xmin(-1);
  bbox.set_ymin(1);
  bbox.set_xmax(1);
  bbox.set_ymax(2);

  NormalizedBBox decode_bbox;
  DecodeBBox(prior_bbox, prior_variance, bbox, &decode_bbox);

  EXPECT_NEAR(decode_bbox.xmin(), 0, eps);
  EXPECT_NEAR(decode_bbox.ymin(), 0.2, eps);
  EXPECT_NEAR(decode_bbox.xmax(), 0.4, eps);
  EXPECT_NEAR(decode_bbox.ymax(), 0.5, eps);
}

TEST_F(BBoxUtilTest, TestDecodeBBoxes) {
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  vector<NormalizedBBox> bboxes;
  for (int i = 1; i < 5; ++i) {
    NormalizedBBox prior_bbox;
    prior_bbox.set_xmin(0.1*i);
    prior_bbox.set_ymin(0.1*i);
    prior_bbox.set_xmax(0.1*i + 0.2);
    prior_bbox.set_ymax(0.1*i + 0.2);
    prior_bboxes.push_back(prior_bbox);

    vector<float> prior_variance(4, 0.1);
    prior_variances.push_back(prior_variance);

    NormalizedBBox bbox;
    bbox.set_xmin(-1 * (i%2));
    bbox.set_ymin((i+1)%2);
    bbox.set_xmax((i+1)%2);
    bbox.set_ymax(i%2);
    bboxes.push_back(bbox);
  }

  vector<NormalizedBBox> decode_bboxes;
  DecodeBBoxes(prior_bboxes, prior_variances, bboxes, &decode_bboxes);
  EXPECT_EQ(decode_bboxes.size(), 4);
  for (int i = 1; i < 5; ++i) {
    EXPECT_NEAR(decode_bboxes[i-1].xmin(), 0.1*i + i%2 * -0.1, eps);
    EXPECT_NEAR(decode_bboxes[i-1].ymin(), 0.1*i + (i+1)%2 * 0.1, eps);
    EXPECT_NEAR(decode_bboxes[i-1].xmax(), 0.1*i + 0.2 + (i+1)%2 * 0.1, eps);
    EXPECT_NEAR(decode_bboxes[i-1].ymax(), 0.1*i + 0.2 + i%2 * 0.1, eps);
  }
}

TEST_F(BBoxUtilTest, TestMatchBBoxLableOneBipartite) {
  vector<NormalizedBBox> gt_bboxes;
  vector<NormalizedBBox> pred_bboxes;

  FillBBoxes(&gt_bboxes, &pred_bboxes);

  int label = 1;
  MatchType match_type = MultiBoxLossParameter_MatchType_BIPARTITE;
  float overlap = -1;

  vector<int> match_indices;
  vector<float> match_overlaps;

  MatchBBox(gt_bboxes, pred_bboxes, label, match_type, overlap,
            &match_indices, &match_overlaps);

  EXPECT_EQ(match_indices.size(), 6);
  EXPECT_EQ(match_overlaps.size(), 6);

  EXPECT_EQ(match_indices[0], 0);
  EXPECT_NEAR(match_overlaps[0], 4./9, eps);
  for (int i = 1; i < 6; ++i) {
    EXPECT_EQ(match_indices[i], -1);
    EXPECT_NEAR(match_overlaps[i], 0, eps);
  }
}

TEST_F(BBoxUtilTest, TestMatchBBoxLableAllBipartite) {
  vector<NormalizedBBox> gt_bboxes;
  vector<NormalizedBBox> pred_bboxes;

  FillBBoxes(&gt_bboxes, &pred_bboxes);

  int label = -1;
  MatchType match_type = MultiBoxLossParameter_MatchType_BIPARTITE;
  float overlap = -1;

  vector<int> match_indices;
  vector<float> match_overlaps;

  MatchBBox(gt_bboxes, pred_bboxes, label, match_type, overlap,
            &match_indices, &match_overlaps);

  EXPECT_EQ(match_indices.size(), 6);
  EXPECT_EQ(match_overlaps.size(), 6);

  EXPECT_EQ(match_indices[0], 0);
  EXPECT_EQ(match_indices[3], 1);
  EXPECT_NEAR(match_overlaps[0], 4./9, eps);
  EXPECT_NEAR(match_overlaps[3], 4./8, eps);
  for (int i = 1; i < 6; ++i) {
    if (i == 0 || i == 3) {
      continue;
    }
    EXPECT_EQ(match_indices[i], -1);
    EXPECT_NEAR(match_overlaps[i], 0, eps);
  }
}

TEST_F(BBoxUtilTest, TestMatchBBoxLableOnePerPrediction) {
  vector<NormalizedBBox> gt_bboxes;
  vector<NormalizedBBox> pred_bboxes;

  FillBBoxes(&gt_bboxes, &pred_bboxes);

  int label = 1;
  MatchType match_type = MultiBoxLossParameter_MatchType_PER_PREDICTION;
  float overlap = 0.3;

  vector<int> match_indices;
  vector<float> match_overlaps;

  MatchBBox(gt_bboxes, pred_bboxes, label, match_type, overlap,
            &match_indices, &match_overlaps);

  EXPECT_EQ(match_indices.size(), 6);
  EXPECT_EQ(match_overlaps.size(), 6);

  EXPECT_EQ(match_indices[0], 0);
  EXPECT_EQ(match_indices[1], 0);
  EXPECT_NEAR(match_overlaps[0], 4./9, eps);
  EXPECT_NEAR(match_overlaps[1], 2./6, eps);
  for (int i = 2; i < 6; ++i) {
    EXPECT_EQ(match_indices[i], -1);
    EXPECT_NEAR(match_overlaps[i], 0, eps);
  }
}

TEST_F(BBoxUtilTest, TestMatchBBoxLableAllPerPrediction) {
  vector<NormalizedBBox> gt_bboxes;
  vector<NormalizedBBox> pred_bboxes;

  FillBBoxes(&gt_bboxes, &pred_bboxes);

  int label = -1;
  MatchType match_type = MultiBoxLossParameter_MatchType_PER_PREDICTION;
  float overlap = 0.3;

  vector<int> match_indices;
  vector<float> match_overlaps;

  MatchBBox(gt_bboxes, pred_bboxes, label, match_type, overlap,
            &match_indices, &match_overlaps);

  EXPECT_EQ(match_indices.size(), 6);
  EXPECT_EQ(match_overlaps.size(), 6);

  EXPECT_EQ(match_indices[0], 0);
  EXPECT_EQ(match_indices[1], 0);
  EXPECT_EQ(match_indices[3], 1);
  EXPECT_NEAR(match_overlaps[0], 4./9, eps);
  EXPECT_NEAR(match_overlaps[1], 2./6, eps);
  EXPECT_NEAR(match_overlaps[3], 4./8, eps);
  for (int i = 2; i < 6; ++i) {
    if (i == 3) {
      continue;
    }
    EXPECT_EQ(match_indices[i], -1);
    EXPECT_NEAR(match_overlaps[i], 0, eps);
  }
}

TEST_F(BBoxUtilTest, TestMatchBBoxLableAllPerPredictionEx) {
  vector<NormalizedBBox> gt_bboxes;
  vector<NormalizedBBox> pred_bboxes;

  FillBBoxes(&gt_bboxes, &pred_bboxes);

  int label = -1;
  MatchType match_type = MultiBoxLossParameter_MatchType_PER_PREDICTION;
  float overlap = 0.001;

  vector<int> match_indices;
  vector<float> match_overlaps;

  MatchBBox(gt_bboxes, pred_bboxes, label, match_type, overlap,
            &match_indices, &match_overlaps);

  EXPECT_EQ(match_indices.size(), 6);
  EXPECT_EQ(match_overlaps.size(), 6);

  EXPECT_EQ(match_indices[0], 0);
  EXPECT_EQ(match_indices[1], 0);
  EXPECT_EQ(match_indices[2], 0);
  EXPECT_EQ(match_indices[3], 1);
  EXPECT_EQ(match_indices[4], 1);
  EXPECT_EQ(match_indices[5], -1);
  EXPECT_NEAR(match_overlaps[0], 4./9, eps);
  EXPECT_NEAR(match_overlaps[1], 2./6, eps);
  EXPECT_NEAR(match_overlaps[2], 2./8, eps);
  EXPECT_NEAR(match_overlaps[3], 4./8, eps);
  EXPECT_NEAR(match_overlaps[4], 1./11, eps);
  EXPECT_NEAR(match_overlaps[5], 0., eps);
}

TEST_F(BBoxUtilTest, TestGetGroundTruth) {
  const int num_gt = 4;
  Blob<float> gt_blob(1, 1, num_gt, 8);
  float* gt_data = gt_blob.mutable_cpu_data();
  for (int i = 0; i < 4; ++i) {
    int image_id = ceil(i / 2.);
    gt_data[i * 8] = image_id;
    gt_data[i * 8 + 1] = i;
    gt_data[i * 8 + 2] = 0;
    gt_data[i * 8 + 3] = 0.1;
    gt_data[i * 8 + 4] = 0.1;
    gt_data[i * 8 + 5] = 0.3;
    gt_data[i * 8 + 6] = 0.3;
    gt_data[i * 8 + 7] = i % 2;
  }

  map<int, vector<NormalizedBBox> > all_gt_bboxes;
  GetGroundTruth(gt_data, num_gt, -1, true, &all_gt_bboxes);

  EXPECT_EQ(all_gt_bboxes.size(), 3);

  EXPECT_EQ(all_gt_bboxes[0].size(), 1);
  EXPECT_EQ(all_gt_bboxes[0][0].label(), 0);
  EXPECT_NEAR(all_gt_bboxes[0][0].xmin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[0][0].ymin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[0][0].xmax(), 0.3, eps);
  EXPECT_NEAR(all_gt_bboxes[0][0].ymax(), 0.3, eps);
  EXPECT_EQ(all_gt_bboxes[0][0].difficult(), false);
  EXPECT_NEAR(all_gt_bboxes[0][0].size(), 0.04, eps);

  EXPECT_EQ(all_gt_bboxes[1].size(), 2);
  for (int i = 1; i < 3; ++i) {
    EXPECT_EQ(all_gt_bboxes[1][i-1].label(), i);
    EXPECT_NEAR(all_gt_bboxes[1][i-1].xmin(), 0.1, eps);
    EXPECT_NEAR(all_gt_bboxes[1][i-1].ymin(), 0.1, eps);
    EXPECT_NEAR(all_gt_bboxes[1][i-1].xmax(), 0.3, eps);
    EXPECT_NEAR(all_gt_bboxes[1][i-1].ymax(), 0.3, eps);
    EXPECT_EQ(all_gt_bboxes[1][i-1].difficult(), i % 2);
    EXPECT_NEAR(all_gt_bboxes[1][i-1].size(), 0.04, eps);
  }

  EXPECT_EQ(all_gt_bboxes[2].size(), 1);
  EXPECT_EQ(all_gt_bboxes[2][0].label(), 3);
  EXPECT_NEAR(all_gt_bboxes[2][0].xmin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[2][0].ymin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[2][0].xmax(), 0.3, eps);
  EXPECT_NEAR(all_gt_bboxes[2][0].ymax(), 0.3, eps);
  EXPECT_EQ(all_gt_bboxes[2][0].difficult(), true);
  EXPECT_NEAR(all_gt_bboxes[2][0].size(), 0.04, eps);

  // Skip difficult ground truth.
  GetGroundTruth(gt_data, num_gt, -1, false, &all_gt_bboxes);

  EXPECT_EQ(all_gt_bboxes.size(), 2);

  EXPECT_EQ(all_gt_bboxes[0].size(), 1);
  EXPECT_EQ(all_gt_bboxes[0][0].label(), 0);
  EXPECT_NEAR(all_gt_bboxes[0][0].xmin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[0][0].ymin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[0][0].xmax(), 0.3, eps);
  EXPECT_NEAR(all_gt_bboxes[0][0].ymax(), 0.3, eps);
  EXPECT_EQ(all_gt_bboxes[0][0].difficult(), false);
  EXPECT_NEAR(all_gt_bboxes[0][0].size(), 0.04, eps);

  EXPECT_EQ(all_gt_bboxes[1].size(), 1);
  EXPECT_EQ(all_gt_bboxes[1][0].label(), 2);
  EXPECT_NEAR(all_gt_bboxes[1][0].xmin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[1][0].ymin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[1][0].xmax(), 0.3, eps);
  EXPECT_NEAR(all_gt_bboxes[1][0].ymax(), 0.3, eps);
  EXPECT_EQ(all_gt_bboxes[1][0].difficult(), false);
  EXPECT_NEAR(all_gt_bboxes[1][0].size(), 0.04, eps);
}

TEST_F(BBoxUtilTest, TestGetGroundTruthLabelBBox) {
  const int num_gt = 4;
  Blob<float> gt_blob(1, 1, num_gt, 8);
  float* gt_data = gt_blob.mutable_cpu_data();
  for (int i = 0; i < 4; ++i) {
    int image_id = ceil(i / 2.);
    gt_data[i * 8] = image_id;
    gt_data[i * 8 + 1] = i;
    gt_data[i * 8 + 2] = 0;
    gt_data[i * 8 + 3] = 0.1;
    gt_data[i * 8 + 4] = 0.1;
    gt_data[i * 8 + 5] = 0.3;
    gt_data[i * 8 + 6] = 0.3;
    gt_data[i * 8 + 7] = i % 2;
  }

  map<int, LabelBBox> all_gt_bboxes;
  GetGroundTruth(gt_data, num_gt, -1, true, &all_gt_bboxes);

  EXPECT_EQ(all_gt_bboxes.size(), 3);

  EXPECT_EQ(all_gt_bboxes[0].size(), 1);
  EXPECT_EQ(all_gt_bboxes[0].find(0)->first, 0);
  EXPECT_NEAR(all_gt_bboxes[0].find(0)->second[0].xmin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[0].find(0)->second[0].ymin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[0].find(0)->second[0].xmax(), 0.3, eps);
  EXPECT_NEAR(all_gt_bboxes[0].find(0)->second[0].ymax(), 0.3, eps);
  EXPECT_EQ(all_gt_bboxes[0].find(0)->second[0].difficult(), false);
  EXPECT_NEAR(all_gt_bboxes[0].find(0)->second[0].size(), 0.04, eps);

  EXPECT_EQ(all_gt_bboxes[1].size(), 2);
  for (int i = 1; i < 3; ++i) {
    EXPECT_EQ(all_gt_bboxes[1].find(i)->first, i);
    EXPECT_NEAR(all_gt_bboxes[1].find(i)->second[0].xmin(), 0.1, eps);
    EXPECT_NEAR(all_gt_bboxes[1].find(i)->second[0].ymin(), 0.1, eps);
    EXPECT_NEAR(all_gt_bboxes[1].find(i)->second[0].xmax(), 0.3, eps);
    EXPECT_NEAR(all_gt_bboxes[1].find(i)->second[0].ymax(), 0.3, eps);
    EXPECT_EQ(all_gt_bboxes[1].find(i)->second[0].difficult(), i % 2);
    EXPECT_NEAR(all_gt_bboxes[1].find(i)->second[0].size(), 0.04, eps);
  }

  EXPECT_EQ(all_gt_bboxes[2].size(), 1);
  EXPECT_EQ(all_gt_bboxes[2].find(3)->first, 3);
  EXPECT_NEAR(all_gt_bboxes[2].find(3)->second[0].xmin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[2].find(3)->second[0].ymin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[2].find(3)->second[0].xmax(), 0.3, eps);
  EXPECT_NEAR(all_gt_bboxes[2].find(3)->second[0].ymax(), 0.3, eps);
  EXPECT_EQ(all_gt_bboxes[2].find(3)->second[0].difficult(), true);
  EXPECT_NEAR(all_gt_bboxes[2].find(3)->second[0].size(), 0.04, eps);

  // Skip difficult ground truth.
  GetGroundTruth(gt_data, num_gt, -1, false, &all_gt_bboxes);

  EXPECT_EQ(all_gt_bboxes.size(), 2);

  EXPECT_EQ(all_gt_bboxes[0].size(), 1);
  EXPECT_EQ(all_gt_bboxes[0].find(0)->first, 0);
  EXPECT_NEAR(all_gt_bboxes[0].find(0)->second[0].xmin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[0].find(0)->second[0].ymin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[0].find(0)->second[0].xmax(), 0.3, eps);
  EXPECT_NEAR(all_gt_bboxes[0].find(0)->second[0].ymax(), 0.3, eps);
  EXPECT_EQ(all_gt_bboxes[0].find(0)->second[0].difficult(), false);
  EXPECT_NEAR(all_gt_bboxes[0].find(0)->second[0].size(), 0.04, eps);

  EXPECT_EQ(all_gt_bboxes[1].size(), 1);
  EXPECT_EQ(all_gt_bboxes[1].find(2)->first, 2);
  EXPECT_NEAR(all_gt_bboxes[1].find(2)->second[0].xmin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[1].find(2)->second[0].ymin(), 0.1, eps);
  EXPECT_NEAR(all_gt_bboxes[1].find(2)->second[0].xmax(), 0.3, eps);
  EXPECT_NEAR(all_gt_bboxes[1].find(2)->second[0].ymax(), 0.3, eps);
  EXPECT_EQ(all_gt_bboxes[1].find(2)->second[0].difficult(), false);
  EXPECT_NEAR(all_gt_bboxes[1].find(2)->second[0].size(), 0.04, eps);
}

TEST_F(BBoxUtilTest, TestGetLocPredictionsShared) {
  const int num = 2;
  const int num_preds_per_class = 2;
  const int num_loc_classes = 1;
  const bool share_location = true;
  const int dim = num_preds_per_class * num_loc_classes * 4;
  Blob<float> loc_blob(num, dim, 1, 1);
  float* loc_data = loc_blob.mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < num_preds_per_class; ++j) {
      int start_idx = i * dim + j * 4;
      loc_data[start_idx] = i * num_preds_per_class * 0.1 + j * 0.1;
      loc_data[start_idx + 1] = i * num_preds_per_class * 0.1 + j * 0.1;
      loc_data[start_idx + 2] = i * num_preds_per_class * 0.1 + j * 0.1 + 0.2;
      loc_data[start_idx + 3] = i * num_preds_per_class * 0.1 + j * 0.1 + 0.2;
    }
  }

  vector<LabelBBox> all_loc_bboxes;
  GetLocPredictions(loc_data, num, num_preds_per_class, num_loc_classes,
                    share_location, &all_loc_bboxes);

  EXPECT_EQ(all_loc_bboxes.size(), num);

  for (int i = 0; i < num; ++i) {
    EXPECT_EQ(all_loc_bboxes[i].size(), 1);
    LabelBBox::iterator it = all_loc_bboxes[i].begin();
    EXPECT_EQ(it->first, -1);
    const vector<NormalizedBBox>& bboxes = it->second;
    EXPECT_EQ(bboxes.size(), num_preds_per_class);
    float start_value = i * num_preds_per_class * 0.1;
    for (int j = 0; j < num_preds_per_class; ++j) {
      EXPECT_EQ(bboxes[j].has_label(), false);
      EXPECT_NEAR(bboxes[j].xmin(), start_value + j * 0.1, eps);
      EXPECT_NEAR(bboxes[j].ymin(), start_value + j * 0.1, eps);
      EXPECT_NEAR(bboxes[j].xmax(), start_value + j * 0.1 + 0.2, eps);
      EXPECT_NEAR(bboxes[j].ymax(), start_value + j * 0.1 + 0.2, eps);
      EXPECT_EQ(bboxes[j].has_size(), false);
    }
  }
}

TEST_F(BBoxUtilTest, TestGetLocPredictionsUnShared) {
  const int num = 2;
  const int num_preds_per_class = 2;
  const int num_loc_classes = 2;
  const bool share_location = false;
  const int dim = num_preds_per_class * num_loc_classes * 4;
  Blob<float> loc_blob(num, dim, 1, 1);
  float* loc_data = loc_blob.mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < num_preds_per_class; ++j) {
      float start_value = (i * num_preds_per_class + j) * num_loc_classes * 0.1;
      for (int c = 0; c < num_loc_classes; ++c) {
        int idx = ((i * num_preds_per_class + j) * num_loc_classes + c) * 4;
        loc_data[idx] = start_value + c * 0.1;
        loc_data[idx + 1] = start_value + c * 0.1;
        loc_data[idx + 2] = start_value + c * 0.1 + 0.2;
        loc_data[idx + 3] = start_value + c * 0.1 + 0.2;
      }
    }
  }

  vector<LabelBBox> all_loc_bboxes;
  GetLocPredictions(loc_data, num, num_preds_per_class, num_loc_classes,
                    share_location, &all_loc_bboxes);

  EXPECT_EQ(all_loc_bboxes.size(), num);

  for (int i = 0; i < num; ++i) {
    EXPECT_EQ(all_loc_bboxes[i].size(), num_loc_classes);
    for (int c = 0; c < num_loc_classes; ++c) {
      LabelBBox::iterator it = all_loc_bboxes[i].find(c);
      EXPECT_EQ(it->first, c);
      const vector<NormalizedBBox>& bboxes = it->second;
      EXPECT_EQ(bboxes.size(), num_preds_per_class);
      for (int j = 0; j < num_preds_per_class; ++j) {
        float start_value =
            (i * num_preds_per_class + j) * num_loc_classes * 0.1;
        EXPECT_EQ(bboxes[j].has_label(), false);
        EXPECT_NEAR(bboxes[j].xmin(), start_value + c * 0.1, eps);
        EXPECT_NEAR(bboxes[j].ymin(), start_value + c * 0.1, eps);
        EXPECT_NEAR(bboxes[j].xmax(), start_value + c * 0.1 + 0.2, eps);
        EXPECT_NEAR(bboxes[j].ymax(), start_value + c * 0.1 + 0.2, eps);
        EXPECT_EQ(bboxes[j].has_size(), false);
      }
    }
  }
}

TEST_F(BBoxUtilTest, TestGetConfidenceScores) {
  const int num = 2;
  const int num_preds_per_class = 2;
  const int num_classes = 2;
  const int dim = num_preds_per_class * num_classes;
  Blob<float> conf_blob(num, dim, 1, 1);
  float* conf_data = conf_blob.mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < num_preds_per_class; ++j) {
      for (int c = 0; c < num_classes; ++c) {
        int idx = (i * num_preds_per_class + j) * num_classes + c;
        conf_data[idx] = idx * 0.1;
      }
    }
  }

  vector<map<int, vector<float> > > all_conf_preds;
  GetConfidenceScores(conf_data, num, num_preds_per_class, num_classes,
                      &all_conf_preds);

  EXPECT_EQ(all_conf_preds.size(), num);

  for (int i = 0; i < num; ++i) {
    EXPECT_EQ(all_conf_preds[i].size(), num_classes);
    for (int c = 0; c < num_classes; ++c) {
      map<int, vector<float> >::iterator it = all_conf_preds[i].find(c);
      EXPECT_EQ(it->first, c);
      const vector<float>& confidences = it->second;
      EXPECT_EQ(confidences.size(), num_preds_per_class);
      for (int j = 0; j < num_preds_per_class; ++j) {
        int idx = (i * num_preds_per_class + j) * num_classes + c;
        EXPECT_NEAR(confidences[j], idx * 0.1, eps);
      }
    }
  }
}

TEST_F(BBoxUtilTest, TestGetPriorBBoxes) {
  const int num_channels = 2;
  const int num_priors = 2;
  const int dim = num_priors * 4;
  Blob<float> prior_blob(1, num_channels, dim, 1);
  float* prior_data = prior_blob.mutable_cpu_data();
  for (int i = 0; i < num_priors; ++i) {
    prior_data[i * 4] = i * 0.1;
    prior_data[i * 4 + 1] = i * 0.1;
    prior_data[i * 4 + 2] = i * 0.1 + 0.2;
    prior_data[i * 4 + 3] = i * 0.1 + 0.1;
    for (int j = 0; j < 4; ++j) {
      prior_data[dim + i * 4 + j]  = 0.1;
    }
  }

  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, num_priors, &prior_bboxes, &prior_variances);

  EXPECT_EQ(prior_bboxes.size(), num_priors);
  EXPECT_EQ(prior_variances.size(), num_priors);

  for (int i = 0; i < num_priors; ++i) {
    EXPECT_NEAR(prior_bboxes[i].xmin(), i * 0.1, eps);
    EXPECT_NEAR(prior_bboxes[i].ymin(), i * 0.1, eps);
    EXPECT_NEAR(prior_bboxes[i].xmax(), i * 0.1 + 0.2, eps);
    EXPECT_NEAR(prior_bboxes[i].ymax(), i * 0.1 + 0.1, eps);
    EXPECT_EQ(prior_variances[i].size(), 4);
    for (int j = 0; j < 4; ++j) {
      EXPECT_NEAR(prior_variances[i][j], 0.1, eps);
    }
  }
}

TEST_F(BBoxUtilTest, TestGetDetectionResults) {
  const int num = 4;
  const int num_det = (1 + num) * num / 2;
  Blob<float> det_blob(1, 1, num_det, 7);
  float* det_data = det_blob.mutable_cpu_data();
  int idx = 0;
  for (int i = 0; i < num; ++i) {
    int image_id = ceil(i / 2.);
    for (int j = 0; j <= i; ++j) {
      det_data[idx * 7] = image_id;
      det_data[idx * 7 + 1] = i;
      det_data[idx * 7 + 2] = 0;
      det_data[idx * 7 + 3] = 0.1 + j * 0.1;
      det_data[idx * 7 + 4] = 0.1 + j * 0.1;
      det_data[idx * 7 + 5] = 0.3 + j * 0.1;
      det_data[idx * 7 + 6] = 0.3 + j * 0.1;
      ++idx;
    }
  }
  CHECK_EQ(idx, num_det);

  map<int, LabelBBox> all_detections;
  GetDetectionResults(det_data, num_det, -1, &all_detections);

  EXPECT_EQ(all_detections.size(), 3);

  EXPECT_EQ(all_detections[0].size(), 1);
  EXPECT_EQ(all_detections[0].find(0)->first, 0);
  EXPECT_EQ(all_detections[0].find(0)->second.size(), 1);
  EXPECT_NEAR(all_detections[0].find(0)->second[0].xmin(), 0.1, eps);
  EXPECT_NEAR(all_detections[0].find(0)->second[0].ymin(), 0.1, eps);
  EXPECT_NEAR(all_detections[0].find(0)->second[0].xmax(), 0.3, eps);
  EXPECT_NEAR(all_detections[0].find(0)->second[0].ymax(), 0.3, eps);
  EXPECT_NEAR(all_detections[0].find(0)->second[0].size(), 0.04, eps);

  EXPECT_EQ(all_detections[1].size(), 2);
  for (int i = 1; i < 3; ++i) {
    EXPECT_EQ(all_detections[1].find(i)->first, i);
    EXPECT_EQ(all_detections[1].find(i)->second.size(), i + 1);
    for (int j = 0; j <= i; ++j) {
      EXPECT_NEAR(all_detections[1].find(i)->second[j].xmin(),
                  0.1 + j * 0.1, eps);
      EXPECT_NEAR(all_detections[1].find(i)->second[j].ymin(),
                  0.1 + j * 0.1, eps);
      EXPECT_NEAR(all_detections[1].find(i)->second[j].xmax(),
                  0.3 + j * 0.1, eps);
      EXPECT_NEAR(all_detections[1].find(i)->second[j].ymax(),
                  0.3 + j * 0.1, eps);
      EXPECT_NEAR(all_detections[1].find(i)->second[j].size(), 0.04, eps);
    }
  }

  EXPECT_EQ(all_detections[2].size(), 1);
  EXPECT_EQ(all_detections[2].find(3)->first, 3);
  EXPECT_EQ(all_detections[2].find(3)->second.size(), 4);
  for (int j = 0; j <= 3; ++j) {
    EXPECT_NEAR(all_detections[2].find(3)->second[j].xmin(),
                0.1 + j * 0.1, eps);
    EXPECT_NEAR(all_detections[2].find(3)->second[j].ymin(),
                0.1 + j * 0.1, eps);
    EXPECT_NEAR(all_detections[2].find(3)->second[j].xmax(),
                0.3 + j * 0.1, eps);
    EXPECT_NEAR(all_detections[2].find(3)->second[j].ymax(),
                0.3 + j * 0.1, eps);
    EXPECT_NEAR(all_detections[2].find(3)->second[j].size(), 0.04, eps);
  }
}

TEST_F(BBoxUtilTest, TestApplyNMS) {
  vector<NormalizedBBox> bboxes;
  vector<float> scores;
  float nms_threshold = 0.3;
  int top_k = -1;
  bool reuse_overlaps = false;
  map<int, map<int, float> > overlaps;
  vector<int> indices;

  // Fill in bboxes and confidences.
  NormalizedBBox bbox;
  bbox.set_xmin(0.1);
  bbox.set_ymin(0.1);
  bbox.set_xmax(0.3);
  bbox.set_ymax(0.3);
  bboxes.push_back(bbox);
  scores.push_back(0.8);

  bbox.set_xmin(0.2);
  bbox.set_ymin(0.1);
  bbox.set_xmax(0.4);
  bbox.set_ymax(0.3);
  bboxes.push_back(bbox);
  scores.push_back(0.7);

  bbox.set_xmin(0.2);
  bbox.set_ymin(0.0);
  bbox.set_xmax(0.4);
  bbox.set_ymax(0.2);
  bboxes.push_back(bbox);
  scores.push_back(0.4);

  bbox.set_xmin(0.1);
  bbox.set_ymin(0.2);
  bbox.set_xmax(0.4);
  bbox.set_ymax(0.4);
  bboxes.push_back(bbox);
  scores.push_back(0.5);

  ApplyNMS(bboxes, scores, nms_threshold, top_k, reuse_overlaps, &overlaps,
           &indices);

  EXPECT_EQ(overlaps.size(), 0);  // reuse_overlaps is false.
  EXPECT_EQ(indices.size(), 3);
  EXPECT_EQ(indices[0], 0);
  EXPECT_EQ(indices[1], 3);
  EXPECT_EQ(indices[2], 2);

  top_k = 2;
  ApplyNMS(bboxes, scores, nms_threshold, top_k, reuse_overlaps, &overlaps,
           &indices);
  EXPECT_EQ(indices.size(), 1);
  EXPECT_EQ(indices[0], 0);

  top_k = 3;
  nms_threshold = 0.2;
  ApplyNMS(bboxes, scores, nms_threshold, top_k, reuse_overlaps, &overlaps,
           &indices);
  EXPECT_EQ(indices.size(), 1);
  EXPECT_EQ(indices[0], 0);

  reuse_overlaps = true;
  ApplyNMS(bboxes, scores, nms_threshold, top_k, reuse_overlaps, &overlaps,
           &indices);
  EXPECT_EQ(overlaps.size(), 1);
  EXPECT_NEAR(overlaps[0][1], 1./3, eps);
  EXPECT_NEAR(overlaps[0][2], 0.0, eps);
  EXPECT_NEAR(overlaps[0][3], 2./8, eps);

  map<int, map<int, float> > old_overlaps = overlaps;
  ApplyNMS(bboxes, scores, nms_threshold, top_k, reuse_overlaps, &overlaps,
           &indices);
  EXPECT_EQ(old_overlaps.size(), overlaps.size());
  for (int i = 1; i <= 3; ++i) {
    EXPECT_NEAR(old_overlaps[0][i], overlaps[0][i], eps);
  }
}

TEST_F(BBoxUtilTest, TestCumSum) {
  vector<pair<float, int> > pairs;
  vector<int> cumsum;

  pairs.push_back(std::make_pair(0.1, 0));
  pairs.push_back(std::make_pair(0.2, 1));
  pairs.push_back(std::make_pair(0.3, 0));

  CumSum(pairs, &cumsum);

  EXPECT_EQ(cumsum.size(), 3);
  EXPECT_EQ(cumsum[0], 0);
  EXPECT_EQ(cumsum[1], 1);
  EXPECT_EQ(cumsum[2], 1);
}

TEST_F(BBoxUtilTest, TestComputeAP) {
  vector<pair<float, int> > tp;
  vector<pair<float, int> > fp;

  tp.push_back(std::make_pair(1.0, 0));
  tp.push_back(std::make_pair(1.0, 1));
  tp.push_back(std::make_pair(0.9, 1));
  tp.push_back(std::make_pair(0.9, 0));
  tp.push_back(std::make_pair(0.8, 1));
  tp.push_back(std::make_pair(0.7, 0));
  tp.push_back(std::make_pair(0.7, 1));
  tp.push_back(std::make_pair(0.6, 0));
  tp.push_back(std::make_pair(0.5, 0));
  tp.push_back(std::make_pair(0.4, 0));
  tp.push_back(std::make_pair(0.4, 1));

  fp.push_back(std::make_pair(1.0, 1));
  fp.push_back(std::make_pair(1.0, 0));
  fp.push_back(std::make_pair(0.9, 0));
  fp.push_back(std::make_pair(0.9, 1));
  fp.push_back(std::make_pair(0.8, 0));
  fp.push_back(std::make_pair(0.7, 1));
  fp.push_back(std::make_pair(0.7, 0));
  fp.push_back(std::make_pair(0.6, 1));
  fp.push_back(std::make_pair(0.5, 1));
  fp.push_back(std::make_pair(0.4, 1));
  fp.push_back(std::make_pair(0.4, 0));

  float eps = 1e-5;
  vector<float> prec, rec;
  float ap;

  ComputeAP(tp, 5, fp, "Integral", &prec, &rec, &ap);

  EXPECT_NEAR(ap, 0.558528, eps);

  EXPECT_EQ(prec.size(), 11);
  EXPECT_NEAR(prec[0], 0.0/1.0, eps);
  EXPECT_NEAR(prec[1], 1.0/2.0, eps);
  EXPECT_NEAR(prec[2], 2.0/3.0, eps);
  EXPECT_NEAR(prec[3], 2.0/4.0, eps);
  EXPECT_NEAR(prec[4], 3.0/5.0, eps);
  EXPECT_NEAR(prec[5], 3.0/6.0, eps);
  EXPECT_NEAR(prec[6], 4.0/7.0, eps);
  EXPECT_NEAR(prec[7], 4.0/8.0, eps);
  EXPECT_NEAR(prec[8], 4.0/9.0, eps);
  EXPECT_NEAR(prec[9], 4.0/10.0, eps);
  EXPECT_NEAR(prec[10], 5.0/11.0, eps);

  EXPECT_EQ(rec.size(), 11);
  EXPECT_NEAR(rec[0], 0.0, eps);
  EXPECT_NEAR(rec[1], 0.2, eps);
  EXPECT_NEAR(rec[2], 0.4, eps);
  EXPECT_NEAR(rec[3], 0.4, eps);
  EXPECT_NEAR(rec[4], 0.6, eps);
  EXPECT_NEAR(rec[5], 0.6, eps);
  EXPECT_NEAR(rec[6], 0.8, eps);
  EXPECT_NEAR(rec[7], 0.8, eps);
  EXPECT_NEAR(rec[8], 0.8, eps);
  EXPECT_NEAR(rec[9], 0.8, eps);
  EXPECT_NEAR(rec[10], 1.0, eps);

  vector<float> prec_old = prec;
  vector<float> rec_old = rec;
  ComputeAP(tp, 5, fp, "11point", &prec, &rec, &ap);

  EXPECT_NEAR(ap, 0.598662, eps);
  EXPECT_EQ(prec.size(), 11);
  EXPECT_EQ(rec.size(), 11);
  for (int i = 0; i < 11; ++i) {
    EXPECT_NEAR(prec_old[i], prec[i], eps);
    EXPECT_NEAR(rec_old[i], rec[i], eps);
  }

  // Cut the last 4 predictions.
  tp.resize(7);
  fp.resize(7);

  ComputeAP(tp, 5, fp, "Integral", &prec, &rec, &ap);

  EXPECT_NEAR(ap, 0.558528 - prec_old.back() * 0.2, eps);
  EXPECT_EQ(prec.size(), 7);
  EXPECT_EQ(rec.size(), 7);
  for (int i = 0; i < 7; ++i) {
    EXPECT_NEAR(prec_old[i], prec[i], eps);
    EXPECT_NEAR(rec_old[i], rec[i], eps);
  }

  ComputeAP(tp, 5, fp, "11point", &prec, &rec, &ap);

  EXPECT_NEAR(ap, 0.598662 - prec_old.back() * 2 / 11., eps);
  EXPECT_EQ(prec.size(), 7);
  EXPECT_EQ(rec.size(), 7);
  for (int i = 0; i < 7; ++i) {
    EXPECT_NEAR(prec_old[i], prec[i], eps);
    EXPECT_NEAR(rec_old[i], rec[i], eps);
  }
}

}  // namespace caffe
