#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/fast_rcnn_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

float epsilon = 1e-4;

namespace caffe {

template <typename TypeParam>
class SimplerNMSLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SimplerNMSLayerTest()
      : rpn_cls_prob_reshape(new Blob<Dtype>()),
        rpn_bbox_pred(new Blob<Dtype>()),
        im_info(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_bottom_vec_.push_back(rpn_cls_prob_reshape);
    blob_bottom_vec_.push_back(rpn_bbox_pred);
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

  virtual ~SimplerNMSLayerTest() {
    delete rpn_cls_prob_reshape;
    delete rpn_bbox_pred;
    delete im_info;
    delete blob_top_;
  }

  Blob<Dtype>* rpn_cls_prob_reshape;
  Blob<Dtype>* rpn_bbox_pred;
  Blob<Dtype>* im_info;

  Blob<Dtype>* const blob_top_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};


TYPED_TEST_CASE(SimplerNMSLayerTest, TestDtypesAndDevices);

static void InitNMSParam(LayerParameter& layer_param)
{
  SimplerNMSParameter* nms_param = layer_param.mutable_simpler_nms_param();

  nms_param->set_max_num_proposals(300);
  nms_param->set_post_nms_topn(160);
  nms_param->add_scale(8.0f);
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
                //feature@00,  ft@01,  ft@10,  ft@11
//not an object, do not care, just fill with zero
    /*anchor0*/    0,           0,      0,      0,
    /*anchor1*/    0,           0,      0,      0,
    /*anchor2*/    0,           0,      0,      0,
//is an object, the following will be really used
    /*anchor0*/    0.9,         0.2,    0.6,    0.01,
    /*anchor1*/    0.75,        0.8,    0.11,   0.25,
    /*anchor2*/    0.34,        0.113,  0.12,   0.87,
};

static float rpn_bbox_pred_data[] = {
            //feature@00,  ft@01,  ft@10,  ft@11
//anchor0:
    /*dx0*/   0.9,        0.01,   0,      0.02,
    /*dy0*/   0.03,        -0.02,  0.1,    0.01,
    /*dx1*/   0.0,        0.08,   -0.045, 0.015,
    /*dy1*/   0.0546,       0.001,  -0.072, 0.032,
//anchor1:
    /*dx0*/   0.02,         0.3,   0.12,   0.022,
    /*dy0*/   0.03,         -0.04,  0.01,   0.001,
    /*dx1*/   0.006,        0.,  0.045,  0.005,
    /*dy1*/   0.0146,       0.031,  0.072,  0.002,
//anchor2:
    /*dx0*/   -0.019,       0.011,   0.04,    2.062,
    /*dy0*/   0.023,        -0.012,  0.01,    1.061,
    /*dx1*/   0.056,        0.018,   -0.0145, 0.0615,
    /*dy1*/   0.0511,       0.0021,  -0.0172, 0.0632,
};

static float rois_ref_data[] = {
    0, 81.5999908, 0,265.599976, 61.5736732,
    0, 158.66507, 116.994881, 252.246948, 304.477112,
    0, 0, 0, 126.400002, 68.8950729,
    0, 0, 0, 74.9451523, 76.78125,
    0, 0, 0, 52.8622932, 104.661682,
    0, 0, 0, 90.3057861, 94.0579376,
};

TYPED_TEST(SimplerNMSLayerTest, SingleBatch) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InitNMSParam(layer_param);
  SimplerNMSLayer<Dtype> layer(layer_param);

  this->rpn_cls_prob_reshape->Reshape(1, 6, 2, 2);
  Dtype* p = this->rpn_cls_prob_reshape->mutable_cpu_data();
  assert(this->rpn_cls_prob_reshape->count() == sizeof(rpn_cls_prob_reshape_data)/sizeof(float));
  for (size_t i = 0; i < this->rpn_cls_prob_reshape->count(); ++i) {
    p[i] = (Dtype)(rpn_cls_prob_reshape_data[i]);
  }

  this->rpn_bbox_pred->Reshape(1, 12, 2, 2);
  p = this->rpn_bbox_pred->mutable_cpu_data();
  assert(this->rpn_bbox_pred->count() == sizeof(rpn_bbox_pred_data)/sizeof(float));
  for (size_t i = 0; i < this->rpn_bbox_pred->count(); ++i) {
    p[i] = (Dtype)(rpn_bbox_pred_data[i]);
  }

  Blob<Dtype>* const top_ref0 = new Blob<Dtype>();
  top_ref0->Reshape(vector<int>{6, 5});
  p = top_ref0->mutable_cpu_data();
  assert(top_ref0->count() == sizeof(rois_ref_data)/sizeof(float));
  for (size_t i = 0; i < top_ref0->count(); ++i) {
    p[i] = (Dtype)(rois_ref_data[i]);
  }

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  CheckBlob(*this->blob_top_, *top_ref0, 0, 0, top_ref0->count(), epsilon);

  delete top_ref0;
}

}  // namespace caffe
