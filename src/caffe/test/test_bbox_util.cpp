#include <vector>

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/util/bbox_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

static const float eps = 1e-5;

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

  EXPECT_EQ(match_indices.size(), 5);
  EXPECT_EQ(match_overlaps.size(), 5);

  EXPECT_EQ(match_indices[0], 0);
  EXPECT_NEAR(match_overlaps[0], 4./9, eps);
  for (int i = 1; i < 5; ++i) {
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

  EXPECT_EQ(match_indices.size(), 5);
  EXPECT_EQ(match_overlaps.size(), 5);

  EXPECT_EQ(match_indices[0], 0);
  EXPECT_EQ(match_indices[3], 1);
  EXPECT_NEAR(match_overlaps[0], 4./9, eps);
  EXPECT_NEAR(match_overlaps[3], 4./8, eps);
  for (int i = 1; i < 5; ++i) {
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

  EXPECT_EQ(match_indices.size(), 5);
  EXPECT_EQ(match_overlaps.size(), 5);

  EXPECT_EQ(match_indices[0], 0);
  EXPECT_EQ(match_indices[1], 0);
  EXPECT_NEAR(match_overlaps[0], 4./9, eps);
  EXPECT_NEAR(match_overlaps[1], 2./6, eps);
  for (int i = 2; i < 5; ++i) {
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

  EXPECT_EQ(match_indices.size(), 5);
  EXPECT_EQ(match_overlaps.size(), 5);

  EXPECT_EQ(match_indices[0], 0);
  EXPECT_EQ(match_indices[1], 0);
  EXPECT_EQ(match_indices[3], 1);
  EXPECT_NEAR(match_overlaps[0], 4./9, eps);
  EXPECT_NEAR(match_overlaps[1], 2./6, eps);
  EXPECT_NEAR(match_overlaps[3], 4./8, eps);
  for (int i = 2; i < 5; ++i) {
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

  EXPECT_EQ(match_indices.size(), 5);
  EXPECT_EQ(match_overlaps.size(), 5);

  EXPECT_EQ(match_indices[0], 0);
  EXPECT_EQ(match_indices[1], 0);
  EXPECT_EQ(match_indices[2], 0);
  EXPECT_EQ(match_indices[3], 1);
  EXPECT_EQ(match_indices[4], 1);
  EXPECT_NEAR(match_overlaps[0], 4./9, eps);
  EXPECT_NEAR(match_overlaps[1], 2./6, eps);
  EXPECT_NEAR(match_overlaps[2], 2./8, eps);
  EXPECT_NEAR(match_overlaps[3], 4./8, eps);
  EXPECT_NEAR(match_overlaps[4], 1./11, eps);
}

}  // namespace caffe
