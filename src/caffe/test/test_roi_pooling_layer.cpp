#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class ROIPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ROIPoolingLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(1, 3, 12, 8)),
        blob_bottom_rois_(new Blob<Dtype>(2, 1, 1, 4)),
        blob_top_data_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    //for (int i = 0; i < blob_bottom_data_->count(); ++i) {
    //  blob_bottom_data_->mutable_cpu_data()[i] = i;
    //}
    blob_bottom_vec_.push_back(blob_bottom_data_);
    int i = 0;
    blob_bottom_rois_->mutable_cpu_data()[0 + 4*i] = 0;
    blob_bottom_rois_->mutable_cpu_data()[1 + 4*i] = 1;
    blob_bottom_rois_->mutable_cpu_data()[2 + 4*i] = 4;
    blob_bottom_rois_->mutable_cpu_data()[3 + 4*i] = 6;
    i = 1;
    blob_bottom_rois_->mutable_cpu_data()[0 + 4*i] = 2;
    blob_bottom_rois_->mutable_cpu_data()[1 + 4*i] = 1;
    blob_bottom_rois_->mutable_cpu_data()[2 + 4*i] = 7;
    blob_bottom_rois_->mutable_cpu_data()[3 + 4*i] = 7;
    /*
    i = 2;
    blob_bottom_rois_->mutable_cpu_data()[0 + 4*i] = 1;
    blob_bottom_rois_->mutable_cpu_data()[1 + 4*i] = 3;
    blob_bottom_rois_->mutable_cpu_data()[2 + 4*i] = 1;
    blob_bottom_rois_->mutable_cpu_data()[3 + 4*i] = 5;
    i = 3;
    blob_bottom_rois_->mutable_cpu_data()[0 + 4*i] = 0;
    blob_bottom_rois_->mutable_cpu_data()[1 + 4*i] = 3;
    blob_bottom_rois_->mutable_cpu_data()[2 + 4*i] = 3;
    blob_bottom_rois_->mutable_cpu_data()[3 + 4*i] = 3;
    */

    blob_bottom_vec_.push_back(blob_bottom_rois_);
    blob_top_vec_.push_back(blob_top_data_);
  }
  virtual ~ROIPoolingLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_rois_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_rois_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ROIPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(ROIPoolingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ROIPoolingParameter* roi_pooling_param =
      layer_param.mutable_roi_pooling_param();
  roi_pooling_param->set_pyramid_height(2);
  roi_pooling_param->set_n_rois(2);
  ROIPoolingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

//TYPED_TEST(SoftmaxWithLossLayerTest, TestGradient) {
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  layer_param.add_loss_weight(3);
//  SoftmaxWithLossLayer<Dtype> layer(layer_param);
//  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
//  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//      this->blob_top_vec_, 0);
//}
//
//TYPED_TEST(PoolingLayerTest, TestGradientMax) {
//  typedef typename TypeParam::Dtype Dtype;
//  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
//    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
//      LayerParameter layer_param;
//      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
//      pooling_param->set_kernel_h(kernel_h);
//      pooling_param->set_kernel_w(kernel_w);
//      pooling_param->set_stride(2);
//      pooling_param->set_pad(1);
//      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
//      PoolingLayer<Dtype> layer(layer_param);
//      GradientChecker<Dtype> checker(1e-4, 1e-2);
//      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//          this->blob_top_vec_);
//    }
//  }
//}

}  // namespace caffe
