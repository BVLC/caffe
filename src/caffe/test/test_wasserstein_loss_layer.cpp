#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/wasserstein_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename Dtype>
class WassersteinLossLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  WassersteinLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 4, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 4, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    PositiveUnitballFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~WassersteinLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(WassersteinLossLayerTest, TestDtypes);


TYPED_TEST(WassersteinLossLayerTest, TestGradientCPU) {
  LayerParameter layer_param;
  layer_param.mutable_wasserstein_param()->set_ground_metric(
    CMAKE_SOURCE_DIR "caffe/test/test_data/wasserstein_ground_metric.h5");
  layer_param.mutable_wasserstein_param()->set_scaling_iter(100);
  layer_param.mutable_wasserstein_param()->set_shift_gradient(true);
  WassersteinLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-4, 2*1e-2, 1701, 0, -1);
  checker.CheckGradientUpToShift(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
