#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/fast_rcnn_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

static float epsilon = 1e-4;

namespace caffe {

template <typename TypeParam>
class FasterRcnnDetectionOutputLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  FasterRcnnDetectionOutputLayerTest()
      : rpn_cls_prob_reshape(new Blob<Dtype>()),
        rpn_bbox_pred(new Blob<Dtype>()),
        rois(new Blob<Dtype>()),
        im_info(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_bottom_vec_.push_back(rpn_bbox_pred);
    blob_bottom_vec_.push_back(rpn_cls_prob_reshape);
    blob_bottom_vec_.push_back(rois);
    blob_bottom_vec_.push_back(im_info);
    blob_top_vec_.push_back(blob_top_);

    im_info->Reshape(1,1,1,3);
    int image_width  = 800;
    int image_height = 600;
    int image_scale  = 1.6;
    im_info->mutable_cpu_data()[1] = image_width;
    im_info->mutable_cpu_data()[0] = image_height;
    im_info->mutable_cpu_data()[2] = image_scale;

    if (std::is_same<Dtype, half_float::half>::value)
      epsilon = 0.5;
  }

  virtual ~FasterRcnnDetectionOutputLayerTest() {
    delete rpn_cls_prob_reshape;
    delete rpn_bbox_pred;
    delete rois;
    delete im_info;
    delete blob_top_;
  }

  Blob<Dtype>* rpn_cls_prob_reshape;
  Blob<Dtype>* rpn_bbox_pred;
  Blob<Dtype>* rois;
  Blob<Dtype>* im_info;

  Blob<Dtype>* const blob_top_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};


TYPED_TEST_CASE(FasterRcnnDetectionOutputLayerTest, TestDtypesAndDevices);

static void InitFasterRcnnDetectionOutputParam(LayerParameter& layer_param)
{
  FasterRcnnDetectionOutputParameter* detection_output_param = layer_param.mutable_faster_rcnn_detection_output_param();

  detection_output_param->set_num_classes(2);
  detection_output_param->set_background_label_id(0);
  detection_output_param->mutable_nms_param()->set_nms_threshold(0.3);
  detection_output_param->set_confidence_threshold(0.8);
}

template <typename Dtype>
static void CheckBlob(const Blob<Dtype>& blob, const Blob<Dtype>& ref,
                      uint_tp start_blob, uint_tp start_ref, uint_tp count, float epsilon) {

  ASSERT_GE(blob.count(), start_blob + count) << "insufficient size of blob";
  ASSERT_GE(ref.count(), start_ref + count) << "insufficient size of ref";

  const Dtype* data     = (const Dtype*)blob.cpu_data() + start_blob;
  const Dtype* ref_data = (const Dtype*)ref.cpu_data() + start_ref;

  for (int_tp i = 0; i < blob.count(); ++i) {
    EXPECT_NEAR(data[i], ref_data[i], epsilon);
  }
}

static float rpn_cls_prob_reshape_data[] = {
    0.2, 0.85,
    0.1, 0.9,
    0.6, 0.4,
};

static float rois_data[] = {
    0, 0, 0, 100, 200,
    0, 10, 10, 90, 200,
    0, 200, 100, 300, 300,
};

static float rpn_bbox_pred_data[] = {
    0.9,    0.01,  0,     0.02,   0.03,  -0.02,  0.1,     0.01,
    -0.019, 0.011, 0.04,  2.062,  0.056, 0.018,  -0.0145, 0.0615,
    0.02,   0.3,   0.12,  0.022,  0.03,  -0.04,  0.01,    0.001,
};

static float top_ref_data[] = {
    0, 1, 0.899999976, 15.1190109, 7.38039398, 94.9529877, 210.495605,
};

TYPED_TEST(FasterRcnnDetectionOutputLayerTest, forward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InitFasterRcnnDetectionOutputParam(layer_param);
  FasterRcnnDetectionOutputLayer<Dtype> layer(layer_param);

  this->rpn_cls_prob_reshape->Reshape(vector<int>{3, 2});
  Dtype* p = this->rpn_cls_prob_reshape->mutable_cpu_data();
  assert(this->rpn_cls_prob_reshape->count() == sizeof(rpn_cls_prob_reshape_data)/sizeof(float));
  for (size_t i = 0; i < this->rpn_cls_prob_reshape->count(); ++i) {
    p[i] = (Dtype)(rpn_cls_prob_reshape_data[i]);
  }

  this->rpn_bbox_pred->Reshape(vector<int>{3, 8});
  p = this->rpn_bbox_pred->mutable_cpu_data();
  assert(this->rpn_bbox_pred->count() == sizeof(rpn_bbox_pred_data)/sizeof(float));
  for (size_t i = 0; i < this->rpn_bbox_pred->count(); ++i) {
    p[i] = (Dtype)(rpn_bbox_pred_data[i]);
  }

  this->rois->Reshape(vector<int>{3, 5});
  p = this->rois->mutable_cpu_data();
  assert(this->rois->count() == sizeof(rois_data)/sizeof(float));
  for (size_t i = 0; i < this->rois->count(); ++i) {
    p[i] = (Dtype)(rois_data[i]);
  }

  Blob<Dtype>* const top_ref0 = new Blob<Dtype>();
  top_ref0->Reshape(1,1,1,7);
  p = top_ref0->mutable_cpu_data();
  assert(top_ref0->count() == sizeof(top_ref_data)/sizeof(float));
  for (size_t i = 0; i < top_ref0->count(); ++i) {
    p[i] = (Dtype)(top_ref_data[i]);
  }

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  CheckBlob(*this->blob_top_, *top_ref0, 0, 0, top_ref0->count(), epsilon);

  delete top_ref0;
}

}  // namespace caffe
