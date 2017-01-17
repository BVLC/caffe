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
  // 4/9 with label 1
  // 0 with label 2
  bbox.set_xmin(0.1);
  bbox.set_ymin(0);
  bbox.set_xmax(0.4);
  bbox.set_ymax(0.3);
  pred_bboxes->push_back(bbox);

  // 2/6 with label 1
  // 0 with label 2
  bbox.set_xmin(0);
  bbox.set_ymin(0.1);
  bbox.set_xmax(0.2);
  bbox.set_ymax(0.3);
  pred_bboxes->push_back(bbox);

  // 2/8 with label 1
  // 1/11 with label 2
  bbox.set_xmin(0.2);
  bbox.set_ymin(0.1);
  bbox.set_xmax(0.4);
  bbox.set_ymax(0.4);
  pred_bboxes->push_back(bbox);

  // 0 with label 1
  // 4/8 with label 2
  bbox.set_xmin(0.4);
  bbox.set_ymin(0.3);
  bbox.set_xmax(0.7);
  bbox.set_ymax(0.5);
  pred_bboxes->push_back(bbox);

  // 0 with label 1
  // 1/11 with label 2
  bbox.set_xmin(0.5);
  bbox.set_ymin(0.4);
  bbox.set_xmax(0.7);
  bbox.set_ymax(0.7);
  pred_bboxes->push_back(bbox);

  // 0 with label 1
  // 0 with label 2
  bbox.set_xmin(0.7);
  bbox.set_ymin(0.7);
  bbox.set_xmax(0.8);
  bbox.set_ymax(0.8);
  pred_bboxes->push_back(bbox);
}

template <typename TypeParam>
class BBoxUtilTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
};

class CPUBBoxUtilTest : public BBoxUtilTest<CPUDevice<float> > {
};

TEST_F(CPUBBoxUtilTest, TestIntersectBBox) {
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

TEST_F(CPUBBoxUtilTest, TestBBoxSize) {
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

TEST_F(CPUBBoxUtilTest, TestScaleBBox) {
  NormalizedBBox bbox;
  bbox.set_xmin(0.21);
  bbox.set_ymin(0.32);
  bbox.set_xmax(0.33);
  bbox.set_ymax(0.54);
  NormalizedBBox scale_bbox;
  float eps = 1e-5;

  int height = 10;
  int width = 20;
  ScaleBBox(bbox, height, width, &scale_bbox);
  EXPECT_NEAR(scale_bbox.xmin(), 4.2, eps);
  EXPECT_NEAR(scale_bbox.ymin(), 3.2, eps);
  EXPECT_NEAR(scale_bbox.xmax(), 6.6, eps);
  EXPECT_NEAR(scale_bbox.ymax(), 5.4, eps);
  EXPECT_NEAR(scale_bbox.size(), 10.88, eps);

  height = 1;
  width = 1;
  ScaleBBox(bbox, height, width, &scale_bbox);
  EXPECT_NEAR(bbox.xmin(), scale_bbox.xmin(), eps);
  EXPECT_NEAR(bbox.ymin(), scale_bbox.ymin(), eps);
  EXPECT_NEAR(bbox.xmax(), scale_bbox.xmax(), eps);
  EXPECT_NEAR(bbox.ymax(), scale_bbox.ymax(), eps);
  EXPECT_NEAR(scale_bbox.size(), 0.0264, eps);
}

TEST_F(CPUBBoxUtilTest, TestClipBBox) {
  NormalizedBBox bbox;
  NormalizedBBox clip_bbox;

  bbox.set_xmin(0.2);
  bbox.set_ymin(0.3);
  bbox.set_xmax(0.3);
  bbox.set_ymax(0.5);
  ClipBBox(bbox, &clip_bbox);
  EXPECT_NEAR(bbox.xmin(), clip_bbox.xmin(), eps);
  EXPECT_NEAR(bbox.ymin(), clip_bbox.ymin(), eps);
  EXPECT_NEAR(bbox.xmax(), clip_bbox.xmax(), eps);
  EXPECT_NEAR(bbox.ymax(), clip_bbox.ymax(), eps);
  EXPECT_NEAR(clip_bbox.size(), 0.02, eps);

  bbox.set_xmin(-0.2);
  bbox.set_ymin(-0.3);
  bbox.set_xmax(1.3);
  bbox.set_ymax(1.5);
  ClipBBox(bbox, &clip_bbox);
  EXPECT_NEAR(clip_bbox.xmin(), 0., eps);
  EXPECT_NEAR(clip_bbox.ymin(), 0., eps);
  EXPECT_NEAR(clip_bbox.xmax(), 1., eps);
  EXPECT_NEAR(clip_bbox.ymax(), 1., eps);
  EXPECT_NEAR(clip_bbox.size(), 1., eps);
}

TEST_F(CPUBBoxUtilTest, TestOutputBBox) {
  NormalizedBBox bbox;
  bbox.set_xmin(-0.1);
  bbox.set_ymin(0.3);
  bbox.set_xmax(0.3);
  bbox.set_ymax(0.5);
  pair<int, int> img_size(300, 500);
  bool has_resize = false;
  ResizeParameter resize_param;
  resize_param.set_height(300);
  resize_param.set_width(300);
  NormalizedBBox out_bbox;

  OutputBBox(bbox, img_size, has_resize, resize_param, &out_bbox);
  CHECK_EQ(out_bbox.xmin(), 0.);
  CHECK_EQ(out_bbox.ymin(), 90.);
  CHECK_EQ(out_bbox.xmax(), 150.);
  CHECK_EQ(out_bbox.ymax(), 150.);

  has_resize = true;
  resize_param.set_resize_mode(ResizeParameter_Resize_mode_WARP);
  OutputBBox(bbox, img_size, has_resize, resize_param, &out_bbox);
  CHECK_EQ(out_bbox.xmin(), 0.);
  CHECK_EQ(out_bbox.ymin(), 90.);
  CHECK_EQ(out_bbox.xmax(), 150.);
  CHECK_EQ(out_bbox.ymax(), 150.);

  resize_param.set_resize_mode(ResizeParameter_Resize_mode_FIT_SMALL_SIZE);
  OutputBBox(bbox, img_size, has_resize, resize_param, &out_bbox);
  CHECK_EQ(out_bbox.xmin(), 0.);
  CHECK_EQ(out_bbox.ymin(), 90.);
  CHECK_EQ(out_bbox.xmax(), 150.);
  CHECK_EQ(out_bbox.ymax(), 150.);

  resize_param.set_resize_mode(ResizeParameter_Resize_mode_FIT_SMALL_SIZE);
  resize_param.set_height_scale(300);
  resize_param.set_width_scale(300);
  OutputBBox(bbox, img_size, has_resize, resize_param, &out_bbox);
  CHECK_EQ(out_bbox.xmin(), 0.);
  CHECK_EQ(out_bbox.ymin(), 90.);
  CHECK_EQ(out_bbox.xmax(), 90.);
  CHECK_EQ(out_bbox.ymax(), 150.);

  resize_param.set_resize_mode(
      ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD);
  OutputBBox(bbox, img_size, has_resize, resize_param, &out_bbox);
  CHECK_EQ(out_bbox.xmin(), 0.);
  CHECK_EQ(out_bbox.ymin(), 50.);
  CHECK_EQ(out_bbox.xmax(), 150.);
  CHECK_EQ(out_bbox.ymax(), 150.);

  img_size.first = 500;
  img_size.second = 300;
  OutputBBox(bbox, img_size, has_resize, resize_param, &out_bbox);
  CHECK_EQ(out_bbox.xmin(), 0.);
  CHECK_EQ(out_bbox.ymin(), 150.);
  CHECK_EQ(out_bbox.xmax(), 50.);
  CHECK_EQ(out_bbox.ymax(), 250.);
}

TEST_F(CPUBBoxUtilTest, TestJaccardOverlap) {
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

TEST_F(CPUBBoxUtilTest, TestEncodeBBoxCorner) {
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

  CodeType code_type = PriorBoxParameter_CodeType_CORNER;
  NormalizedBBox encode_bbox;

  bool encode_variance_in_target = true;
  EncodeBBox(prior_bbox, prior_variance, code_type, encode_variance_in_target,
             bbox, &encode_bbox);
  EXPECT_NEAR(encode_bbox.xmin(), -0.1, eps);
  EXPECT_NEAR(encode_bbox.ymin(), 0.1, eps);
  EXPECT_NEAR(encode_bbox.xmax(), 0.1, eps);
  EXPECT_NEAR(encode_bbox.ymax(), 0.2, eps);

  encode_variance_in_target = false;
  EncodeBBox(prior_bbox, prior_variance, code_type, encode_variance_in_target,
             bbox, &encode_bbox);
  EXPECT_NEAR(encode_bbox.xmin(), -1, eps);
  EXPECT_NEAR(encode_bbox.ymin(), 1, eps);
  EXPECT_NEAR(encode_bbox.xmax(), 1, eps);
  EXPECT_NEAR(encode_bbox.ymax(), 2, eps);
}

TEST_F(CPUBBoxUtilTest, TestEncodeBBoxCenterSize) {
  NormalizedBBox prior_bbox;
  prior_bbox.set_xmin(0.1);
  prior_bbox.set_ymin(0.1);
  prior_bbox.set_xmax(0.3);
  prior_bbox.set_ymax(0.3);
  vector<float> prior_variance;
  prior_variance.push_back(0.1);
  prior_variance.push_back(0.1);
  prior_variance.push_back(0.2);
  prior_variance.push_back(0.2);

  NormalizedBBox bbox;
  bbox.set_xmin(0);
  bbox.set_ymin(0.2);
  bbox.set_xmax(0.4);
  bbox.set_ymax(0.5);

  CodeType code_type = PriorBoxParameter_CodeType_CENTER_SIZE;
  NormalizedBBox encode_bbox;

  bool encode_variance_in_target = true;
  EncodeBBox(prior_bbox, prior_variance, code_type, encode_variance_in_target,
             bbox, &encode_bbox);
  EXPECT_NEAR(encode_bbox.xmin(), 0, eps);
  EXPECT_NEAR(encode_bbox.ymin(), 0.75, eps);
  EXPECT_NEAR(encode_bbox.xmax(), log(2.), eps);
  EXPECT_NEAR(encode_bbox.ymax(), log(3./2), eps);

  encode_variance_in_target = false;
  EncodeBBox(prior_bbox, prior_variance, code_type, encode_variance_in_target,
             bbox, &encode_bbox);
  float eps = 1e-5;
  EXPECT_NEAR(encode_bbox.xmin(), 0 / 0.1, eps);
  EXPECT_NEAR(encode_bbox.ymin(), 0.75 / 0.1, eps);
  EXPECT_NEAR(encode_bbox.xmax(), log(2.) / 0.2, eps);
  EXPECT_NEAR(encode_bbox.ymax(), log(3./2) / 0.2, eps);
}

TEST_F(CPUBBoxUtilTest, TestDecodeBBoxCorner) {
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

  CodeType code_type = PriorBoxParameter_CodeType_CORNER;
  NormalizedBBox decode_bbox;

  bool variance_encoded_in_target = false;
  DecodeBBox(prior_bbox, prior_variance, code_type, variance_encoded_in_target,
             false, bbox, &decode_bbox);
  EXPECT_NEAR(decode_bbox.xmin(), 0, eps);
  EXPECT_NEAR(decode_bbox.ymin(), 0.2, eps);
  EXPECT_NEAR(decode_bbox.xmax(), 0.4, eps);
  EXPECT_NEAR(decode_bbox.ymax(), 0.5, eps);

  variance_encoded_in_target = true;
  DecodeBBox(prior_bbox, prior_variance, code_type, variance_encoded_in_target,
             false, bbox, &decode_bbox);
  EXPECT_NEAR(decode_bbox.xmin(), -0.9, eps);
  EXPECT_NEAR(decode_bbox.ymin(), 1.1, eps);
  EXPECT_NEAR(decode_bbox.xmax(), 1.3, eps);
  EXPECT_NEAR(decode_bbox.ymax(), 2.3, eps);
}

TEST_F(CPUBBoxUtilTest, TestDecodeBBoxCenterSize) {
  NormalizedBBox prior_bbox;
  prior_bbox.set_xmin(0.1);
  prior_bbox.set_ymin(0.1);
  prior_bbox.set_xmax(0.3);
  prior_bbox.set_ymax(0.3);
  vector<float> prior_variance;
  prior_variance.push_back(0.1);
  prior_variance.push_back(0.1);
  prior_variance.push_back(0.2);
  prior_variance.push_back(0.2);

  NormalizedBBox bbox;
  bbox.set_xmin(0);
  bbox.set_ymin(0.75);
  bbox.set_xmax(log(2));
  bbox.set_ymax(log(3./2));

  CodeType code_type = PriorBoxParameter_CodeType_CENTER_SIZE;
  NormalizedBBox decode_bbox;

  bool variance_encoded_in_target = true;
  DecodeBBox(prior_bbox, prior_variance, code_type, variance_encoded_in_target,
             false, bbox, &decode_bbox);
  EXPECT_NEAR(decode_bbox.xmin(), 0, eps);
  EXPECT_NEAR(decode_bbox.ymin(), 0.2, eps);
  EXPECT_NEAR(decode_bbox.xmax(), 0.4, eps);
  EXPECT_NEAR(decode_bbox.ymax(), 0.5, eps);

  bbox.set_xmin(0);
  bbox.set_ymin(7.5);
  bbox.set_xmax(log(2) * 5);
  bbox.set_ymax(log(3./2) * 5);
  variance_encoded_in_target = false;
  DecodeBBox(prior_bbox, prior_variance, code_type, variance_encoded_in_target,
             false, bbox, &decode_bbox);
  EXPECT_NEAR(decode_bbox.xmin(), 0, eps);
  EXPECT_NEAR(decode_bbox.ymin(), 0.2, eps);
  EXPECT_NEAR(decode_bbox.xmax(), 0.4, eps);
  EXPECT_NEAR(decode_bbox.ymax(), 0.5, eps);
}

TEST_F(CPUBBoxUtilTest, TestDecodeBBoxesCorner) {
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

  CodeType code_type = PriorBoxParameter_CodeType_CORNER;
  vector<NormalizedBBox> decode_bboxes;

  bool variance_encoded_in_target = false;
  DecodeBBoxes(prior_bboxes, prior_variances, code_type,
               variance_encoded_in_target, false, bboxes, &decode_bboxes);
  EXPECT_EQ(decode_bboxes.size(), 4);
  for (int i = 1; i < 5; ++i) {
    EXPECT_NEAR(decode_bboxes[i-1].xmin(), 0.1*i + i%2 * -0.1, eps);
    EXPECT_NEAR(decode_bboxes[i-1].ymin(), 0.1*i + (i+1)%2 * 0.1, eps);
    EXPECT_NEAR(decode_bboxes[i-1].xmax(), 0.1*i + 0.2 + (i+1)%2 * 0.1, eps);
    EXPECT_NEAR(decode_bboxes[i-1].ymax(), 0.1*i + 0.2 + i%2 * 0.1, eps);
  }

  variance_encoded_in_target = true;
  DecodeBBoxes(prior_bboxes, prior_variances, code_type,
               variance_encoded_in_target, false, bboxes, &decode_bboxes);
  EXPECT_EQ(decode_bboxes.size(), 4);
  for (int i = 1; i < 5; ++i) {
    EXPECT_NEAR(decode_bboxes[i-1].xmin(), 0.1*i + i%2 * -1, eps);
    EXPECT_NEAR(decode_bboxes[i-1].ymin(), 0.1*i + (i+1)%2, eps);
    EXPECT_NEAR(decode_bboxes[i-1].xmax(), 0.1*i + 0.2 + (i+1)%2, eps);
    EXPECT_NEAR(decode_bboxes[i-1].ymax(), 0.1*i + 0.2 + i%2, eps);
  }
}

TEST_F(CPUBBoxUtilTest, TestDecodeBBoxesCenterSize) {
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

    vector<float> prior_variance;
    prior_variance.push_back(0.1);
    prior_variance.push_back(0.1);
    prior_variance.push_back(0.2);
    prior_variance.push_back(0.2);
    prior_variances.push_back(prior_variance);

    NormalizedBBox bbox;
    bbox.set_xmin(0);
    bbox.set_ymin(0.75);
    bbox.set_xmax(log(2.));
    bbox.set_ymax(log(3./2));
    bboxes.push_back(bbox);
  }

  CodeType code_type = PriorBoxParameter_CodeType_CENTER_SIZE;
  vector<NormalizedBBox> decode_bboxes;

  bool variance_encoded_in_target = true;
  DecodeBBoxes(prior_bboxes, prior_variances, code_type,
               variance_encoded_in_target, false, bboxes, &decode_bboxes);
  EXPECT_EQ(decode_bboxes.size(), 4);
  float eps = 1e-5;
  for (int i = 1; i < 5; ++i) {
    EXPECT_NEAR(decode_bboxes[i-1].xmin(), 0 + (i - 1) * 0.1, eps);
    EXPECT_NEAR(decode_bboxes[i-1].ymin(), 0.2 + (i - 1) * 0.1, eps);
    EXPECT_NEAR(decode_bboxes[i-1].xmax(), 0.4 + (i - 1) * 0.1, eps);
    EXPECT_NEAR(decode_bboxes[i-1].ymax(), 0.5 + (i - 1) * 0.1, eps);
  }

  variance_encoded_in_target = false;
  for (int i = 0; i < 4; ++i) {
    NormalizedBBox bbox;
    bboxes[i].set_xmin(0);
    bboxes[i].set_ymin(7.5);
    bboxes[i].set_xmax(log(2.) * 5);
    bboxes[i].set_ymax(log(3./2) * 5);
  }
  DecodeBBoxes(prior_bboxes, prior_variances, code_type,
               variance_encoded_in_target, false, bboxes, &decode_bboxes);
  EXPECT_EQ(decode_bboxes.size(), 4);
  for (int i = 1; i < 5; ++i) {
    EXPECT_NEAR(decode_bboxes[i-1].xmin(), 0 + (i - 1) * 0.1, eps);
    EXPECT_NEAR(decode_bboxes[i-1].ymin(), 0.2 + (i - 1) * 0.1, eps);
    EXPECT_NEAR(decode_bboxes[i-1].xmax(), 0.4 + (i - 1) * 0.1, eps);
    EXPECT_NEAR(decode_bboxes[i-1].ymax(), 0.5 + (i - 1) * 0.1, eps);
  }
}

TEST_F(CPUBBoxUtilTest, TestMatchBBoxLableOneBipartite) {
  vector<NormalizedBBox> gt_bboxes;
  vector<NormalizedBBox> pred_bboxes;

  FillBBoxes(&gt_bboxes, &pred_bboxes);

  int label = 1;
  MatchType match_type = MultiBoxLossParameter_MatchType_BIPARTITE;
  float overlap = -1;

  vector<int> match_indices;
  vector<float> match_overlaps;

  MatchBBox(gt_bboxes, pred_bboxes, label, match_type, overlap, true,
            &match_indices, &match_overlaps);

  EXPECT_EQ(match_indices.size(), 6);
  EXPECT_EQ(match_overlaps.size(), 6);

  EXPECT_EQ(match_indices[0], 0);
  EXPECT_EQ(match_indices[1], -1);
  EXPECT_EQ(match_indices[2], -1);
  EXPECT_NEAR(match_overlaps[0], 4./9, eps);
  EXPECT_NEAR(match_overlaps[1], 2./6, eps);
  EXPECT_NEAR(match_overlaps[2], 2./8, eps);
  for (int i = 3; i < 6; ++i) {
    EXPECT_EQ(match_indices[i], -1);
    EXPECT_NEAR(match_overlaps[i], 0, eps);
  }
}

TEST_F(CPUBBoxUtilTest, TestMatchBBoxLableAllBipartite) {
  vector<NormalizedBBox> gt_bboxes;
  vector<NormalizedBBox> pred_bboxes;

  FillBBoxes(&gt_bboxes, &pred_bboxes);

  int label = -1;
  MatchType match_type = MultiBoxLossParameter_MatchType_BIPARTITE;
  float overlap = -1;

  vector<int> match_indices;
  vector<float> match_overlaps;

  MatchBBox(gt_bboxes, pred_bboxes, label, match_type, overlap, true,
            &match_indices, &match_overlaps);

  EXPECT_EQ(match_indices.size(), 6);
  EXPECT_EQ(match_overlaps.size(), 6);

  EXPECT_EQ(match_indices[0], 0);
  EXPECT_EQ(match_indices[3], 1);
  EXPECT_NEAR(match_overlaps[0], 4./9, eps);
  EXPECT_NEAR(match_overlaps[1], 2./6, eps);
  EXPECT_NEAR(match_overlaps[2], 2./8, eps);
  EXPECT_NEAR(match_overlaps[3], 4./8, eps);
  EXPECT_NEAR(match_overlaps[4], 1./11, eps);
  EXPECT_NEAR(match_overlaps[5], 0., eps);
  for (int i = 0; i < 6; ++i) {
    if (i == 0 || i == 3) {
      continue;
    }
    EXPECT_EQ(match_indices[i], -1);
  }
}

TEST_F(CPUBBoxUtilTest, TestMatchBBoxLableOnePerPrediction) {
  vector<NormalizedBBox> gt_bboxes;
  vector<NormalizedBBox> pred_bboxes;

  FillBBoxes(&gt_bboxes, &pred_bboxes);

  int label = 1;
  MatchType match_type = MultiBoxLossParameter_MatchType_PER_PREDICTION;
  float overlap = 0.3;

  vector<int> match_indices;
  vector<float> match_overlaps;

  MatchBBox(gt_bboxes, pred_bboxes, label, match_type, overlap, true,
            &match_indices, &match_overlaps);

  EXPECT_EQ(match_indices.size(), 6);
  EXPECT_EQ(match_overlaps.size(), 6);

  EXPECT_EQ(match_indices[0], 0);
  EXPECT_EQ(match_indices[1], 0);
  EXPECT_EQ(match_indices[2], -1);
  EXPECT_NEAR(match_overlaps[0], 4./9, eps);
  EXPECT_NEAR(match_overlaps[1], 2./6, eps);
  EXPECT_NEAR(match_overlaps[2], 2./8, eps);
  for (int i = 3; i < 6; ++i) {
    EXPECT_EQ(match_indices[i], -1);
    EXPECT_NEAR(match_overlaps[i], 0, eps);
  }
}

TEST_F(CPUBBoxUtilTest, TestMatchBBoxLableAllPerPrediction) {
  vector<NormalizedBBox> gt_bboxes;
  vector<NormalizedBBox> pred_bboxes;

  FillBBoxes(&gt_bboxes, &pred_bboxes);

  int label = -1;
  MatchType match_type = MultiBoxLossParameter_MatchType_PER_PREDICTION;
  float overlap = 0.3;

  vector<int> match_indices;
  vector<float> match_overlaps;

  MatchBBox(gt_bboxes, pred_bboxes, label, match_type, overlap, true,
            &match_indices, &match_overlaps);

  EXPECT_EQ(match_indices.size(), 6);
  EXPECT_EQ(match_overlaps.size(), 6);

  EXPECT_EQ(match_indices[0], 0);
  EXPECT_EQ(match_indices[1], 0);
  EXPECT_EQ(match_indices[2], -1);
  EXPECT_EQ(match_indices[3], 1);
  EXPECT_EQ(match_indices[4], -1);
  EXPECT_EQ(match_indices[5], -1);
  EXPECT_NEAR(match_overlaps[0], 4./9, eps);
  EXPECT_NEAR(match_overlaps[1], 2./6, eps);
  EXPECT_NEAR(match_overlaps[2], 2./8, eps);
  EXPECT_NEAR(match_overlaps[3], 4./8, eps);
  EXPECT_NEAR(match_overlaps[4], 1./11, eps);
  EXPECT_NEAR(match_overlaps[5], 0, eps);
}

TEST_F(CPUBBoxUtilTest, TestMatchBBoxLableAllPerPredictionEx) {
  vector<NormalizedBBox> gt_bboxes;
  vector<NormalizedBBox> pred_bboxes;

  FillBBoxes(&gt_bboxes, &pred_bboxes);

  int label = -1;
  MatchType match_type = MultiBoxLossParameter_MatchType_PER_PREDICTION;
  float overlap = 0.001;

  vector<int> match_indices;
  vector<float> match_overlaps;

  MatchBBox(gt_bboxes, pred_bboxes, label, match_type, overlap, true,
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

TEST_F(CPUBBoxUtilTest, TestGetGroundTruth) {
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

TEST_F(CPUBBoxUtilTest, TestGetGroundTruthLabelBBox) {
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

TEST_F(CPUBBoxUtilTest, TestGetLocPredictionsShared) {
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

TEST_F(CPUBBoxUtilTest, TestGetLocPredictionsUnShared) {
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

TEST_F(CPUBBoxUtilTest, TestGetConfidenceScores) {
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

TEST_F(CPUBBoxUtilTest, TestComputeConfLoss) {
  const int num = 2;
  const int num_preds_per_class = 2;
  const int num_classes = 2;
  const int dim = num_preds_per_class * num_classes;
  Blob<float> conf_blob(num, dim, 1, 1);
  float* conf_data = conf_blob.mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
    int sign = i % 2 ? 1 : -1;
    for (int j = 0; j < num_preds_per_class; ++j) {
      for (int c = 0; c < num_classes; ++c) {
        int idx = (i * num_preds_per_class + j) * num_classes + c;
        conf_data[idx] = sign * idx * 0.1;
      }
    }
  }

  vector<vector<float> > all_conf_loss;
  ConfLossType loss_type = MultiBoxLossParameter_ConfLossType_LOGISTIC;
  ComputeConfLoss(conf_data, num, num_preds_per_class, num_classes,
                  -1, loss_type, &all_conf_loss);

  EXPECT_EQ(all_conf_loss.size(), num);
  EXPECT_EQ(all_conf_loss[0].size(), num_preds_per_class);
  EXPECT_NEAR(all_conf_loss[0][0],
              -(log(exp(0.)/(1.+exp(0.))) + log(exp(0.1)/(1+exp(0.1)))), eps);
  EXPECT_NEAR(all_conf_loss[0][1],
              -(log(exp(0.2)/(1.+exp(0.2))) + log(exp(0.3)/(1+exp(0.3)))), eps);
  EXPECT_EQ(all_conf_loss[1].size(), num_preds_per_class);
  EXPECT_NEAR(all_conf_loss[1][0],
              -(log(exp(-0.4)/(1.+exp(-0.4))) + log(exp(-0.5)/(1+exp(-0.5)))),
              eps);
  EXPECT_NEAR(all_conf_loss[1][1],
              -(log(exp(-0.6)/(1.+exp(-0.6))) + log(exp(-0.7)/(1+exp(-0.7)))),
              eps);

  ComputeConfLoss(conf_data, num, num_preds_per_class, num_classes,
                  0, loss_type, &all_conf_loss);

  EXPECT_EQ(all_conf_loss.size(), num);
  EXPECT_EQ(all_conf_loss[0].size(), num_preds_per_class);
  EXPECT_NEAR(all_conf_loss[0][0],
              -(log(1./(1.+exp(0.))) + log(exp(0.1)/(1+exp(0.1)))), eps);
  EXPECT_NEAR(all_conf_loss[0][1],
              -(log(1./(1.+exp(0.2))) + log(exp(0.3)/(1+exp(0.3)))), eps);
  EXPECT_EQ(all_conf_loss[1].size(), num_preds_per_class);
  EXPECT_NEAR(all_conf_loss[1][0],
              -(log(1./(1.+exp(-0.4))) + log(exp(-0.5)/(1+exp(-0.5)))), eps);
  EXPECT_NEAR(all_conf_loss[1][1],
              -(log(1./(1.+exp(-0.6))) + log(exp(-0.7)/(1+exp(-0.7)))), eps);

  loss_type = MultiBoxLossParameter_ConfLossType_SOFTMAX;
  ComputeConfLoss(conf_data, num, num_preds_per_class, num_classes,
                  0, loss_type, &all_conf_loss);

  EXPECT_EQ(all_conf_loss.size(), num);
  for (int i = 0; i < num; ++i) {
    EXPECT_EQ(all_conf_loss[i].size(), num_preds_per_class);
    int sign = i % 2 ? 1 : -1;
    for (int j = 0; j < num_preds_per_class; ++j) {
      if (sign == 1) {
        EXPECT_NEAR(all_conf_loss[i][j], -log(exp(-0.1)/(1+exp(-0.1))), eps);
      } else {
        EXPECT_NEAR(all_conf_loss[i][j], -log(1./(1+exp(-0.1))), eps);
      }
    }
  }
}

TEST_F(CPUBBoxUtilTest, TestComputeConfLossMatch) {
  const int num = 2;
  const int num_preds_per_class = 2;
  const int num_classes = 2;
  const int dim = num_preds_per_class * num_classes;
  Blob<float> conf_blob(num, dim, 1, 1);
  float* conf_data = conf_blob.mutable_cpu_data();
  vector<map<int, vector<int> > > all_match_indices;
  map<int, vector<NormalizedBBox> > all_gt_bboxes;
  for (int i = 0; i < num; ++i) {
    int sign = i % 2 ? 1 : -1;
    for (int j = 0; j < num_preds_per_class; ++j) {
      for (int c = 0; c < num_classes; ++c) {
        int idx = (i * num_preds_per_class + j) * num_classes + c;
        conf_data[idx] = sign * idx * 0.1;
      }
    }
    map<int, vector<int> > match_indices;
    vector<int> indices(num_preds_per_class, -1);
    match_indices[-1] = indices;
    if (i == 1) {
      NormalizedBBox gt_bbox;
      gt_bbox.set_label(1);
      all_gt_bboxes[i].push_back(gt_bbox);
      // The first prior in second image is matched to a gt bbox of label 1.
      match_indices[-1][0] = 0;
    }
    all_match_indices.push_back(match_indices);
  }

  vector<vector<float> > all_conf_loss;
  ConfLossType loss_type = MultiBoxLossParameter_ConfLossType_LOGISTIC;
  ComputeConfLoss(conf_data, num, num_preds_per_class, num_classes,
                  -1, loss_type, all_match_indices, all_gt_bboxes,
                  &all_conf_loss);

  EXPECT_EQ(all_conf_loss.size(), num);
  EXPECT_EQ(all_conf_loss[0].size(), num_preds_per_class);
  EXPECT_NEAR(all_conf_loss[0][0],
              -(log(exp(0.)/(1.+exp(0.))) + log(exp(0.1)/(1+exp(0.1)))), eps);
  EXPECT_NEAR(all_conf_loss[0][1],
              -(log(exp(0.2)/(1.+exp(0.2))) + log(exp(0.3)/(1+exp(0.3)))), eps);
  EXPECT_EQ(all_conf_loss[1].size(), num_preds_per_class);
  EXPECT_NEAR(all_conf_loss[1][0],
              -(log(exp(-0.4)/(1.+exp(-0.4))) + log(1./(1+exp(-0.5)))),
              eps);
  EXPECT_NEAR(all_conf_loss[1][1],
              -(log(exp(-0.6)/(1.+exp(-0.6))) + log(exp(-0.7)/(1+exp(-0.7)))),
              eps);

  ComputeConfLoss(conf_data, num, num_preds_per_class, num_classes,
                  0, loss_type, all_match_indices, all_gt_bboxes,
                  &all_conf_loss);

  EXPECT_EQ(all_conf_loss.size(), num);
  EXPECT_EQ(all_conf_loss[0].size(), num_preds_per_class);
  EXPECT_NEAR(all_conf_loss[0][0],
              -(log(1./(1.+exp(0.))) + log(exp(0.1)/(1+exp(0.1)))), eps);
  EXPECT_NEAR(all_conf_loss[0][1],
              -(log(1./(1.+exp(0.2))) + log(exp(0.3)/(1+exp(0.3)))), eps);
  EXPECT_EQ(all_conf_loss[1].size(), num_preds_per_class);
  EXPECT_NEAR(all_conf_loss[1][0],
              -(log(exp(-0.4)/(1.+exp(-0.4))) + log(1./(1+exp(-0.5)))), eps);
  EXPECT_NEAR(all_conf_loss[1][1],
              -(log(1./(1.+exp(-0.6))) + log(exp(-0.7)/(1+exp(-0.7)))), eps);

  loss_type = MultiBoxLossParameter_ConfLossType_SOFTMAX;
  ComputeConfLoss(conf_data, num, num_preds_per_class, num_classes,
                  0, loss_type, all_match_indices, all_gt_bboxes,
                  &all_conf_loss);

  EXPECT_EQ(all_conf_loss.size(), num);
  for (int i = 0; i < num; ++i) {
    EXPECT_EQ(all_conf_loss[i].size(), num_preds_per_class);
    int sign = i % 2 ? 1 : -1;
    for (int j = 0; j < num_preds_per_class; ++j) {
      if (sign == 1) {
        if (j == 0) {
          EXPECT_NEAR(all_conf_loss[i][j], -log(1./(1+exp(-0.1))), eps);
        } else {
          EXPECT_NEAR(all_conf_loss[i][j], -log(exp(-0.1)/(1+exp(-0.1))), eps);
        }
      } else {
        EXPECT_NEAR(all_conf_loss[i][j], -log(1./(1+exp(-0.1))), eps);
      }
    }
  }
}

TEST_F(CPUBBoxUtilTest, TestGetPriorBBoxes) {
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

TEST_F(CPUBBoxUtilTest, TestGetDetectionResults) {
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

TEST_F(CPUBBoxUtilTest, TestApplyNMS) {
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

TEST_F(CPUBBoxUtilTest, TestApplyNMSFast) {
  vector<NormalizedBBox> bboxes;
  vector<float> scores;
  float score_threshold = 0.;
  float nms_threshold = 0.3;
  float eta = 1.;
  int top_k = -1;
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

  ApplyNMSFast(bboxes, scores, score_threshold, nms_threshold, eta, top_k,
               &indices);

  EXPECT_EQ(indices.size(), 3);
  EXPECT_EQ(indices[0], 0);
  EXPECT_EQ(indices[1], 3);
  EXPECT_EQ(indices[2], 2);

  top_k = 2;
  ApplyNMSFast(bboxes, scores, score_threshold, nms_threshold, eta, top_k,
               &indices);
  EXPECT_EQ(indices.size(), 1);
  EXPECT_EQ(indices[0], 0);

  top_k = 3;
  nms_threshold = 0.2;
  ApplyNMSFast(bboxes, scores, score_threshold, nms_threshold, eta, top_k,
               &indices);
  EXPECT_EQ(indices.size(), 1);
  EXPECT_EQ(indices[0], 0);

  top_k = -1;
  score_threshold = 0.5;
  ApplyNMSFast(bboxes, scores, score_threshold, nms_threshold, eta, top_k,
               &indices);
  EXPECT_EQ(indices.size(), 1);
  EXPECT_EQ(indices[0], 0);
}

TEST_F(CPUBBoxUtilTest, TestCumSum) {
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

TEST_F(CPUBBoxUtilTest, TestComputeAP) {
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
  ComputeAP(tp, 5, fp, "MaxIntegral", &prec, &rec, &ap);

  EXPECT_NEAR(ap, 0.591861, eps);
  EXPECT_EQ(prec.size(), 11);
  EXPECT_EQ(rec.size(), 11);
  for (int i = 0; i < 11; ++i) {
    EXPECT_NEAR(prec_old[i], prec[i], eps);
    EXPECT_NEAR(rec_old[i], rec[i], eps);
  }

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

  ComputeAP(tp, 5, fp, "MaxIntegral", &prec, &rec, &ap);

  EXPECT_NEAR(ap, 0.591861 - prec_old.back() * 0.2, eps);
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

#ifndef CPU_ONLY
template <typename Dtype>
void FillBBoxes(Dtype* gt_bboxes, Dtype* pred_bboxes) {
}

template <typename Dtype>
class GPUBBoxUtilTest : public BBoxUtilTest<GPUDevice<Dtype> > {
};

TYPED_TEST_CASE(GPUBBoxUtilTest, TestDtypes);

TYPED_TEST(GPUBBoxUtilTest, TestBBoxSize) {
  float size;
  Blob<TypeParam> bbox(1, 1, 1, 4);
  TypeParam* bbox_data = bbox.mutable_cpu_data();

  // Valid box.
  bbox_data[0] = 0.2;
  bbox_data[1] = 0.3;
  bbox_data[2] = 0.3;
  bbox_data[3] = 0.5;
  size = BBoxSizeGPU(bbox_data);
  EXPECT_NEAR(size, 0.02, eps);

  // A line.
  bbox_data[2] = 0.2;
  size = BBoxSizeGPU(bbox_data);
  EXPECT_NEAR(size, 0., eps);

  // Invalid box.
  bbox_data[2] = 0.1;
  size = BBoxSizeGPU(bbox_data);
  EXPECT_NEAR(size, 0., eps);
}

TYPED_TEST(GPUBBoxUtilTest, TestJaccardOverlap) {
  float overlap;
  Blob<TypeParam> bbox1(1, 1, 1, 4);
  TypeParam* bbox1_data = bbox1.mutable_cpu_data();
  bbox1_data[0] = 0.2;
  bbox1_data[1] = 0.3;
  bbox1_data[2] = 0.3;
  bbox1_data[3] = 0.5;

  Blob<TypeParam> bbox2(1, 1, 1, 4);
  TypeParam* bbox2_data = bbox2.mutable_cpu_data();

  // Partially overlapped.
  bbox2_data[0] = 0.1;
  bbox2_data[1] = 0.1;
  bbox2_data[2] = 0.3;
  bbox2_data[3] = 0.4;
  overlap = JaccardOverlapGPU(bbox1_data, bbox2_data);
  EXPECT_NEAR(overlap, 1./7, eps);

  // Fully contain.
  bbox2_data[0] = 0.1;
  bbox2_data[1] = 0.1;
  bbox2_data[2] = 0.4;
  bbox2_data[3] = 0.6;
  overlap = JaccardOverlapGPU(bbox1_data, bbox2_data);
  EXPECT_NEAR(overlap, 2./15, eps);

  // Outside.
  bbox2_data[0] = 0.;
  bbox2_data[1] = 0.;
  bbox2_data[2] = 0.1;
  bbox2_data[3] = 0.1;
  overlap = JaccardOverlapGPU(bbox1_data, bbox2_data);
  EXPECT_NEAR(overlap, 0., eps);
}

TYPED_TEST(GPUBBoxUtilTest, TestDecodeBBoxesCorner) {
  int num = 4;
  Blob<TypeParam> prior_bboxes(1, 2, num * 4, 1);
  TypeParam* prior_data = prior_bboxes.mutable_cpu_data();
  Blob<TypeParam> loc_preds(1, num * 4, 1, 1);
  TypeParam* loc_data = loc_preds.mutable_cpu_data();
  for (int i = 1; i <= num; ++i) {
    prior_data[(i - 1) * 4] = 0.1 * i;
    prior_data[(i - 1) * 4 + 1] = 0.1 * i;
    prior_data[(i - 1) * 4 + 2] = 0.1 * i + 0.2;
    prior_data[(i - 1) * 4 + 3] = 0.1 * i + 0.2;
    for (int j = 0; j < 4; ++j) {
      prior_data[num * 4 + (i - 1) * 4 + j] = 0.1;
    }

    loc_data[(i - 1) * 4] = -1 * (i % 2);
    loc_data[(i - 1) * 4 + 1] = ((i + 1) % 2);
    loc_data[(i - 1) * 4 + 2] = ((i + 1) % 2);
    loc_data[(i - 1) * 4 + 3] = i % 2;
  }

  CodeType code_type = PriorBoxParameter_CodeType_CORNER;
  Blob<TypeParam> bboxes(1, num * 4, 1, 1);
  TypeParam* bbox_data = bboxes.mutable_gpu_data();

  bool variance_encoded_in_target = false;
  DecodeBBoxesGPU(num * 4, loc_data, prior_data, code_type,
                  variance_encoded_in_target, num, false, 1, -1, false,
                  bbox_data);
  TypeParam* bbox_cpu_data = bboxes.mutable_cpu_data();
  for (int i = 1; i <= num; ++i) {
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4], 0.1*i + i%2 * -0.1, eps);
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4 + 1], 0.1*i + (i+1)%2 * 0.1, eps);
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4 + 2],
                0.1*i + 0.2 + (i+1)%2 * 0.1, eps);
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4 + 3], 0.1*i + 0.2 + i%2 * 0.1, eps);
  }

  variance_encoded_in_target = true;
  bbox_data = bboxes.mutable_gpu_data();
  DecodeBBoxesGPU(num * 4, loc_data, prior_data, code_type,
                  variance_encoded_in_target, num, false, 1, -1, false,
                  bbox_data);
  bbox_cpu_data = bboxes.mutable_cpu_data();
  for (int i = 1; i <= num; ++i) {
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4], 0.1*i + i%2 * -1, eps);
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4 + 1], 0.1*i + (i+1)%2, eps);
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4 + 2], 0.1*i + 0.2 + (i+1)%2, eps);
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4 + 3], 0.1*i + 0.2 + i%2, eps);
  }
}

TYPED_TEST(GPUBBoxUtilTest, TestDecodeBBoxesCornerTwoClasses) {
  int num = 4;
  int num_loc_classes = 2;
  Blob<TypeParam> prior_bboxes(1, 2, num * 4, 1);
  TypeParam* prior_data = prior_bboxes.mutable_cpu_data();
  Blob<TypeParam> loc_preds(1, num * num_loc_classes * 4, 1, 1);
  TypeParam* loc_data = loc_preds.mutable_cpu_data();
  for (int i = 1; i <= num; ++i) {
    prior_data[(i - 1) * 4] = 0.1 * i;
    prior_data[(i - 1) * 4 + 1] = 0.1 * i;
    prior_data[(i - 1) * 4 + 2] = 0.1 * i + 0.2;
    prior_data[(i - 1) * 4 + 3] = 0.1 * i + 0.2;
    for (int j = 0; j < 4; ++j) {
      prior_data[num * 4 + (i - 1) * 4 + j] = 0.1;
    }

    for (int j = 0; j < num_loc_classes; ++j) {
      loc_data[((i - 1) * 2 + j) * 4] = -1 * (i % 2) * (2 - j);
      loc_data[((i - 1) * 2 + j) * 4 + 1] = ((i + 1) % 2) * (2 - j);
      loc_data[((i - 1) * 2 + j) * 4 + 2] = ((i + 1) % 2) * (2 - j);
      loc_data[((i - 1) * 2 + j) * 4 + 3] = i % 2 * (2 - j);
    }
  }

  CodeType code_type = PriorBoxParameter_CodeType_CORNER;
  Blob<TypeParam> bboxes(1, num * num_loc_classes * 4, 1, 1);
  TypeParam* bbox_data = bboxes.mutable_gpu_data();

  bool variance_encoded_in_target = false;
  DecodeBBoxesGPU(num * num_loc_classes * 4, loc_data, prior_data, code_type,
                  variance_encoded_in_target, num, false, num_loc_classes, -1,
                  false, bbox_data);
  TypeParam* bbox_cpu_data = bboxes.mutable_cpu_data();
  for (int i = 1; i <= num; ++i) {
    for (int j = 0; j < num_loc_classes; ++j) {
      EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4],
                  0.1*i + i%2 * (2-j) * -0.1, eps);
      EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4 + 1],
                  0.1*i + (i+1)%2 * (2-j) * 0.1, eps);
      EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4 + 2],
                  0.1*i + 0.2 + (i+1)%2 * (2-j) * 0.1, eps);
      EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4 + 3],
                  0.1*i + 0.2 + i%2 * (2-j) * 0.1, eps);
    }
  }

  variance_encoded_in_target = true;
  bbox_data = bboxes.mutable_gpu_data();
  DecodeBBoxesGPU(num * num_loc_classes * 4, loc_data, prior_data, code_type,
                  variance_encoded_in_target, num, false, num_loc_classes, -1,
                  false, bbox_data);
  bbox_cpu_data = bboxes.mutable_cpu_data();
  for (int i = 1; i <= num; ++i) {
    for (int j = 0; j < num_loc_classes; ++j) {
      EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4],
                  0.1*i + i%2 * (2-j) * -1, eps);
      EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4 + 1],
                  0.1*i + (i+1)%2 * (2-j), eps);
      EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4 + 2],
                  0.1*i + 0.2 + (i+1)%2 * (2-j), eps);
      EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4 + 3],
                  0.1*i + 0.2 + i%2 * (2-j), eps);
    }
  }
}

TYPED_TEST(GPUBBoxUtilTest, TestDecodeBBoxesCornerTwoClassesNegClass0) {
  int num = 4;
  int num_loc_classes = 2;
  Blob<TypeParam> prior_bboxes(1, 2, num * 4, 1);
  TypeParam* prior_data = prior_bboxes.mutable_cpu_data();
  Blob<TypeParam> loc_preds(1, num * num_loc_classes * 4, 1, 1);
  TypeParam* loc_data = loc_preds.mutable_cpu_data();
  for (int i = 1; i <= num; ++i) {
    prior_data[(i - 1) * 4] = 0.1 * i;
    prior_data[(i - 1) * 4 + 1] = 0.1 * i;
    prior_data[(i - 1) * 4 + 2] = 0.1 * i + 0.2;
    prior_data[(i - 1) * 4 + 3] = 0.1 * i + 0.2;
    for (int j = 0; j < 4; ++j) {
      prior_data[num * 4 + (i - 1) * 4 + j] = 0.1;
    }

    for (int j = 0; j < num_loc_classes; ++j) {
      loc_data[((i - 1) * 2 + j) * 4] = -1 * (i % 2) * (2 - j);
      loc_data[((i - 1) * 2 + j) * 4 + 1] = ((i + 1) % 2) * (2 - j);
      loc_data[((i - 1) * 2 + j) * 4 + 2] = ((i + 1) % 2) * (2 - j);
      loc_data[((i - 1) * 2 + j) * 4 + 3] = i % 2 * (2 - j);
    }
  }

  CodeType code_type = PriorBoxParameter_CodeType_CORNER;
  Blob<TypeParam> bboxes(1, num * num_loc_classes * 4, 1, 1);
  TypeParam* bbox_data = bboxes.mutable_gpu_data();

  bool variance_encoded_in_target = false;
  DecodeBBoxesGPU(num * num_loc_classes * 4, loc_data, prior_data, code_type,
                  variance_encoded_in_target, num, false, num_loc_classes, 0,
                  false, bbox_data);
  TypeParam* bbox_cpu_data = bboxes.mutable_cpu_data();
  for (int i = 1; i <= num; ++i) {
    for (int j = 0; j < num_loc_classes; ++j) {
      if (j == 0) {
        for (int k = 0; k < 4; ++k) {
          EXPECT_NEAR(bbox_cpu_data[(i - 1) * 2 * 4 + k], 0., eps);
        }
      } else {
        EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4],
                    0.1*i + i%2 * -0.1, eps);
        EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4 + 1],
                    0.1*i + (i+1)%2 * 0.1, eps);
        EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4 + 2],
                    0.1*i + 0.2 + (i+1)%2 * 0.1, eps);
        EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4 + 3],
                    0.1*i + 0.2 + i%2 * 0.1, eps);
      }
    }
  }

  variance_encoded_in_target = true;
  bbox_data = bboxes.mutable_gpu_data();
  DecodeBBoxesGPU(num * num_loc_classes * 4, loc_data, prior_data, code_type,
                  variance_encoded_in_target, num, false, num_loc_classes, 0,
                  false, bbox_data);
  bbox_cpu_data = bboxes.mutable_cpu_data();
  for (int i = 1; i <= num; ++i) {
    for (int j = 0; j < num_loc_classes; ++j) {
      if (j == 0) {
        for (int k = 0; k < 4; ++k) {
          EXPECT_NEAR(bbox_cpu_data[(i - 1) * 2 * 4 + k], 0., eps);
        }
      } else {
        EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4],
                    0.1*i + i%2 * -1, eps);
        EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4 + 1],
                    0.1*i + (i+1)%2, eps);
        EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4 + 2],
                    0.1*i + 0.2 + (i+1)%2, eps);
        EXPECT_NEAR(bbox_cpu_data[((i - 1) * 2 + j) * 4 + 3],
                    0.1*i + 0.2 + i%2, eps);
      }
    }
  }
}

TYPED_TEST(GPUBBoxUtilTest, TestDecodeBBoxesCenterSize) {
  int num = 2;
  Blob<TypeParam> prior_bboxes(1, 2, num * 4, 1);
  TypeParam* prior_data = prior_bboxes.mutable_cpu_data();
  Blob<TypeParam> loc_preds(1, num * 4, 1, 1);
  TypeParam* loc_data = loc_preds.mutable_cpu_data();
  for (int i = 1; i <= num; ++i) {
    prior_data[(i - 1) * 4] = 0.1 * i;
    prior_data[(i - 1) * 4 + 1] = 0.1 * i;
    prior_data[(i - 1) * 4 + 2] = 0.1 * i + 0.2;
    prior_data[(i - 1) * 4 + 3] = 0.1 * i + 0.2;
    prior_data[num * 4 + (i - 1) * 4] = 0.1;
    prior_data[num * 4 + (i - 1) * 4 + 1] = 0.1;
    prior_data[num * 4 + (i - 1) * 4 + 2] = 0.2;
    prior_data[num * 4 + (i - 1) * 4 + 3] = 0.2;

    loc_data[(i - 1) * 4] = 0;
    loc_data[(i - 1) * 4 + 1] = 0.75;
    loc_data[(i - 1) * 4 + 2] = log(2.);
    loc_data[(i - 1) * 4 + 3] = log(3./2);
  }

  CodeType code_type = PriorBoxParameter_CodeType_CENTER_SIZE;
  Blob<TypeParam> bboxes(1, num * 4, 1, 1);
  TypeParam* bbox_data = bboxes.mutable_gpu_data();

  bool variance_encoded_in_target = true;
  DecodeBBoxesGPU(num * 4, loc_data, prior_data, code_type,
                  variance_encoded_in_target, num, false, 1, -1, false,
                  bbox_data);
  TypeParam* bbox_cpu_data = bboxes.mutable_cpu_data();
  for (int i = 1; i <= num; ++i) {
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4], 0 + (i-1) * 0.1, eps);
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4 + 1], 0.2 + (i-1) * 0.1, eps);
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4 + 2], 0.4 + (i-1) * 0.1, eps);
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4 + 3], 0.5 + (i-1) * 0.1, eps);
  }

  variance_encoded_in_target = false;
  for (int i = 1; i <= num; ++i) {
    loc_data[(i - 1) * 4] = 0;
    loc_data[(i - 1) * 4 + 1] = 7.5;
    loc_data[(i - 1) * 4 + 2] = log(2.) * 5;
    loc_data[(i - 1) * 4 + 3] = log(3./2) * 5;
  }
  bbox_data = bboxes.mutable_gpu_data();
  DecodeBBoxesGPU(num * 4, loc_data, prior_data, code_type,
                  variance_encoded_in_target, num, false, 1, -1, false,
                  bbox_data);
  bbox_cpu_data = bboxes.mutable_cpu_data();
  for (int i = 1; i <= num; ++i) {
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4], 0 + (i-1) * 0.1, eps);
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4 + 1], 0.2 + (i-1) * 0.1, eps);
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4 + 2], 0.4 + (i-1) * 0.1, eps);
    EXPECT_NEAR(bbox_cpu_data[(i - 1) * 4 + 3], 0.5 + (i-1) * 0.1, eps);
  }
}

TYPED_TEST(GPUBBoxUtilTest, TestComputeOverlapped) {
  const int num = 2;
  const int num_bboxes = 2;
  const int num_loc_classes = 1;
  const TypeParam overlap_threshold = 0.3;

  // Fill bboxes.
  Blob<TypeParam> bboxes(num, num_bboxes * num_loc_classes * 4, 1, 1);
  TypeParam* bbox_data = bboxes.mutable_cpu_data();
  // image1
  // bbox1
  bbox_data[0] = 0.1;
  bbox_data[1] = 0.1;
  bbox_data[2] = 0.3;
  bbox_data[3] = 0.3;
  // bbox2
  bbox_data[4] = 0.2;
  bbox_data[5] = 0.1;
  bbox_data[6] = 0.4;
  bbox_data[7] = 0.3;
  // image2
  // bbox1
  bbox_data[8] = 0.2;
  bbox_data[9] = 0.0;
  bbox_data[10] = 0.4;
  bbox_data[11] = 0.2;
  // bbox2
  bbox_data[12] = 0.2;
  bbox_data[13] = 0.1;
  bbox_data[14] = 0.4;
  bbox_data[15] = 0.3;

  Blob<bool> overlapped(num, num_loc_classes, num_bboxes, num_bboxes);
  const int total_bboxes = overlapped.count();
  bool* overlapped_data = overlapped.mutable_gpu_data();
  ComputeOverlappedGPU(total_bboxes, bbox_data, num_bboxes, num_loc_classes,
                       overlap_threshold, overlapped_data);
  const bool* overlapped_cpu_data = overlapped.cpu_data();
  // image1
  // bbox1 with all other bboxes
  EXPECT_EQ(overlapped_cpu_data[0], 0);
  EXPECT_EQ(overlapped_cpu_data[1], 1);
  // bbox2 with all other bboxes
  EXPECT_EQ(overlapped_cpu_data[2], 1);
  EXPECT_EQ(overlapped_cpu_data[3], 0);
  // image2
  // bbox1 with all other bboxes
  EXPECT_EQ(overlapped_cpu_data[4], 0);
  EXPECT_EQ(overlapped_cpu_data[5], 1);
  // bbox2 with all other bboxes
  EXPECT_EQ(overlapped_cpu_data[6], 1);
  EXPECT_EQ(overlapped_cpu_data[7], 0);
}

TYPED_TEST(GPUBBoxUtilTest, TestComputeOverlappedMultiClass) {
  const int num = 2;
  const int num_bboxes = 2;
  const int num_loc_classes = 2;
  const TypeParam overlap_threshold = 0.3;

  // Fill bboxes.
  Blob<TypeParam> bboxes(num, num_bboxes * num_loc_classes * 4, 1, 1);
  TypeParam* bbox_data = bboxes.mutable_cpu_data();
  // image1
  // bbox1
  // class1
  bbox_data[0] = 0.1;
  bbox_data[1] = 0.1;
  bbox_data[2] = 0.3;
  bbox_data[3] = 0.3;
  // class2
  bbox_data[4] = 0.0;
  bbox_data[5] = 0.1;
  bbox_data[6] = 0.2;
  bbox_data[7] = 0.3;
  // bbox2
  // class1
  bbox_data[8] = 0.2;
  bbox_data[9] = 0.1;
  bbox_data[10] = 0.4;
  bbox_data[11] = 0.3;
  // class2
  bbox_data[12] = 0.2;
  bbox_data[13] = 0.1;
  bbox_data[14] = 0.4;
  bbox_data[15] = 0.3;
  // image2
  // bbox1
  // class1
  bbox_data[16] = 0.2;
  bbox_data[17] = 0.0;
  bbox_data[18] = 0.4;
  bbox_data[19] = 0.2;
  // class2
  bbox_data[20] = 0.2;
  bbox_data[21] = 0.1;
  bbox_data[22] = 0.4;
  bbox_data[23] = 0.3;
  // bbox2
  // class1
  bbox_data[24] = 0.1;
  bbox_data[25] = 0.1;
  bbox_data[26] = 0.3;
  bbox_data[27] = 0.3;
  // class2
  bbox_data[28] = 0.1;
  bbox_data[29] = 0.1;
  bbox_data[30] = 0.3;
  bbox_data[31] = 0.3;

  Blob<bool> overlapped(num, num_loc_classes, num_bboxes, num_bboxes);
  const int total_bboxes = overlapped.count();
  bool* overlapped_data = overlapped.mutable_gpu_data();
  ComputeOverlappedGPU(total_bboxes, bbox_data, num_bboxes, num_loc_classes,
                       overlap_threshold, overlapped_data);
  const bool* overlapped_cpu_data = overlapped.cpu_data();
  // image1
  // class1
  // bbox1 with all other bboxes
  EXPECT_EQ(overlapped_cpu_data[0], 0);
  EXPECT_EQ(overlapped_cpu_data[1], 1);
  // bbox2 with all other bboxes
  EXPECT_EQ(overlapped_cpu_data[2], 1);
  EXPECT_EQ(overlapped_cpu_data[3], 0);
  // class2
  // bbox1 with all other bboxes
  EXPECT_EQ(overlapped_cpu_data[4], 0);
  EXPECT_EQ(overlapped_cpu_data[5], 0);
  // bbox2 with all other bboxes
  EXPECT_EQ(overlapped_cpu_data[6], 0);
  EXPECT_EQ(overlapped_cpu_data[7], 0);
  // image2
  // class1
  // bbox1 with all other bboxes
  EXPECT_EQ(overlapped_cpu_data[8], 0);
  EXPECT_EQ(overlapped_cpu_data[9], 0);
  // bbox2 with all other bboxes
  EXPECT_EQ(overlapped_cpu_data[10], 0);
  EXPECT_EQ(overlapped_cpu_data[11], 0);
  // class2
  // bbox1 with all other bboxes
  EXPECT_EQ(overlapped_cpu_data[12], 0);
  EXPECT_EQ(overlapped_cpu_data[13], 1);
  // bbox2 with all other bboxes
  EXPECT_EQ(overlapped_cpu_data[14], 1);
  EXPECT_EQ(overlapped_cpu_data[15], 0);
}

TYPED_TEST(GPUBBoxUtilTest, TestSoftMaxGPU) {
  const int num = 2;
  const int num_preds = 2;
  const int num_classes = 2;
  Blob<TypeParam> data_blob(num, num_preds * num_classes, 1, 1);
  Blob<TypeParam> prob_blob(num, num_preds * num_classes, 1, 1);
  TypeParam* cpu_data = data_blob.mutable_cpu_data();
  cpu_data[0] = 0.1;
  cpu_data[1] = 0.9;
  cpu_data[2] = 0.9;
  cpu_data[3] = 0.1;
  cpu_data[4] = 0.3;
  cpu_data[5] = 0.7;
  cpu_data[6] = 0.7;
  cpu_data[7] = 0.3;

  const TypeParam* gpu_data = data_blob.gpu_data();
  TypeParam* gpu_prob = prob_blob.mutable_gpu_data();
  SoftMaxGPU(gpu_data, num * num_preds, num_classes, 1, gpu_prob);

  const TypeParam* cpu_prob = prob_blob.cpu_data();
  EXPECT_NEAR(cpu_prob[0], exp(-0.8) / (exp(-0.8) + 1), eps);
  EXPECT_NEAR(cpu_prob[1], 1 / (exp(-0.8) + 1), eps);
  EXPECT_NEAR(cpu_prob[2], 1 / (exp(-0.8) + 1), eps);
  EXPECT_NEAR(cpu_prob[3], exp(-0.8) / (exp(-0.8) + 1), eps);
  EXPECT_NEAR(cpu_prob[4], exp(-0.4) / (exp(-0.4) + 1), eps);
  EXPECT_NEAR(cpu_prob[5], 1 / (exp(-0.4) + 1), eps);
  EXPECT_NEAR(cpu_prob[6], 1 / (exp(-0.4) + 1), eps);
  EXPECT_NEAR(cpu_prob[7], exp(-0.4) / (exp(-0.4) + 1), eps);
}

TYPED_TEST(GPUBBoxUtilTest, TestComputeConfLossMatchGPU) {
  const int num = 2;
  const int num_preds_per_class = 2;
  const int num_classes = 2;
  const int dim = num_preds_per_class * num_classes;
  Blob<TypeParam> conf_blob(num, dim, 1, 1);
  TypeParam* conf_data = conf_blob.mutable_cpu_data();
  vector<map<int, vector<int> > > all_match_indices;
  map<int, vector<NormalizedBBox> > all_gt_bboxes;
  for (int i = 0; i < num; ++i) {
    int sign = i % 2 ? 1 : -1;
    for (int j = 0; j < num_preds_per_class; ++j) {
      for (int c = 0; c < num_classes; ++c) {
        int idx = (i * num_preds_per_class + j) * num_classes + c;
        conf_data[idx] = sign * idx * 0.1;
      }
    }
    map<int, vector<int> > match_indices;
    vector<int> indices(num_preds_per_class, -1);
    match_indices[-1] = indices;
    if (i == 1) {
      NormalizedBBox gt_bbox;
      gt_bbox.set_label(1);
      all_gt_bboxes[i].push_back(gt_bbox);
      // The first prior in second image is matched to a gt bbox of label 1.
      match_indices[-1][0] = 0;
    }
    all_match_indices.push_back(match_indices);
  }

  vector<vector<float> > all_conf_loss;
  ConfLossType loss_type = MultiBoxLossParameter_ConfLossType_LOGISTIC;
  ComputeConfLossGPU(conf_blob, num, num_preds_per_class, num_classes,
      -1, loss_type, all_match_indices, all_gt_bboxes, &all_conf_loss);

  EXPECT_EQ(all_conf_loss.size(), num);
  EXPECT_EQ(all_conf_loss[0].size(), num_preds_per_class);
  EXPECT_NEAR(all_conf_loss[0][0],
              -(log(exp(0.)/(1.+exp(0.))) + log(exp(0.1)/(1+exp(0.1)))), eps);
  EXPECT_NEAR(all_conf_loss[0][1],
              -(log(exp(0.2)/(1.+exp(0.2))) + log(exp(0.3)/(1+exp(0.3)))), eps);
  EXPECT_EQ(all_conf_loss[1].size(), num_preds_per_class);
  EXPECT_NEAR(all_conf_loss[1][0],
              -(log(exp(-0.4)/(1.+exp(-0.4))) + log(1./(1+exp(-0.5)))),
              eps);
  EXPECT_NEAR(all_conf_loss[1][1],
              -(log(exp(-0.6)/(1.+exp(-0.6))) + log(exp(-0.7)/(1+exp(-0.7)))),
              eps);

  ComputeConfLossGPU(conf_blob, num, num_preds_per_class, num_classes,
      0, loss_type, all_match_indices, all_gt_bboxes, &all_conf_loss);

  EXPECT_EQ(all_conf_loss.size(), num);
  EXPECT_EQ(all_conf_loss[0].size(), num_preds_per_class);
  EXPECT_NEAR(all_conf_loss[0][0],
              -(log(1./(1.+exp(0.))) + log(exp(0.1)/(1+exp(0.1)))), eps);
  EXPECT_NEAR(all_conf_loss[0][1],
              -(log(1./(1.+exp(0.2))) + log(exp(0.3)/(1+exp(0.3)))), eps);
  EXPECT_EQ(all_conf_loss[1].size(), num_preds_per_class);
  EXPECT_NEAR(all_conf_loss[1][0],
              -(log(exp(-0.4)/(1.+exp(-0.4))) + log(1./(1+exp(-0.5)))), eps);
  EXPECT_NEAR(all_conf_loss[1][1],
              -(log(1./(1.+exp(-0.6))) + log(exp(-0.7)/(1+exp(-0.7)))), eps);

  loss_type = MultiBoxLossParameter_ConfLossType_SOFTMAX;
  ComputeConfLossGPU(conf_blob, num, num_preds_per_class, num_classes,
      0, loss_type, all_match_indices, all_gt_bboxes, &all_conf_loss);

  EXPECT_EQ(all_conf_loss.size(), num);
  for (int i = 0; i < num; ++i) {
    EXPECT_EQ(all_conf_loss[i].size(), num_preds_per_class);
    int sign = i % 2 ? 1 : -1;
    for (int j = 0; j < num_preds_per_class; ++j) {
      if (sign == 1) {
        if (j == 0) {
          EXPECT_NEAR(all_conf_loss[i][j], -log(1./(1+exp(-0.1))), eps);
        } else {
          EXPECT_NEAR(all_conf_loss[i][j], -log(exp(-0.1)/(1+exp(-0.1))), eps);
        }
      } else {
        EXPECT_NEAR(all_conf_loss[i][j], -log(1./(1+exp(-0.1))), eps);
      }
    }
  }
}

#endif

}  // namespace caffe
