#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/fast_rcnn_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

const float epsilon = 1e-4;

namespace caffe {

template <typename TypeParam>
class ROIPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ROIPoolingLayerTest()
      : conv(new Blob<Dtype>()),
        rois(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_bottom_vec_.push_back(conv);
    blob_bottom_vec_.push_back(rois);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ROIPoolingLayerTest() {
    delete conv;
    delete rois;
    delete blob_top_;
  }

  Blob<Dtype>* conv;
  Blob<Dtype>* rois;

  Blob<Dtype>* const blob_top_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  };


TYPED_TEST_CASE(ROIPoolingLayerTest, TestDtypesAndDevices);

static void InitROIPoolingParam(LayerParameter& layer_param)
{
  ROIPoolingParameter* roi_param = layer_param.mutable_roi_pooling_param();

  roi_param->set_pooled_w(3);
  roi_param->set_pooled_h(2);
  roi_param->set_spatial_scale(1);
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

TYPED_TEST(ROIPoolingLayerTest, Forward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InitROIPoolingParam(layer_param);
  ROIPoolingLayer<Dtype> layer(layer_param);

  this->conv->Reshape(1, 1, 60, 80);
  Dtype* p = this->conv->mutable_cpu_data();
  for (size_t i = 0; i < this->conv->count(); ++i) {
    p[i] = (Dtype)(i);
  }

  this->rois->Reshape(vector<int>{128, 5});
  p = this->rois->mutable_cpu_data();
  for (size_t i = 0; i < this->rois->count(); i = i+5) {
    p[i] = (Dtype)(0);
    p[i+1] = (Dtype)(1);
    p[i+2] = (Dtype)(1);
    p[i+3] = (Dtype)(6);
    p[i+4] = (Dtype)(4);
  }

  Blob<Dtype>* const top_ref0 = new Blob<Dtype>();
  top_ref0->Reshape(128, 1, 3, 2);
  p = top_ref0->mutable_cpu_data();
  for (size_t i = 0; i < top_ref0->count(); i = i + 6) {
    p[i] = (Dtype)(162);
    p[i+1] = (Dtype)(164);
    p[i+2] = (Dtype)(166);
    p[i+3] = (Dtype)(322);
    p[i+4] = (Dtype)(324);
    p[i+5] = (Dtype)(326);
  }

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  CheckBlob(*this->blob_top_, *top_ref0, 0, 0, top_ref0->count(), epsilon);

  delete top_ref0;
}

}  // namespace caffe
