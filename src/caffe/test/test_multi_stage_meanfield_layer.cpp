#include <cstring>
#include <vector>
#include <boost/timer.hpp>

#include "gtest/gtest.h"

#include "caffe/util/tvg_util.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {


template <typename TypeParam>
class MultiStageMeanfieldLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MultiStageMeanfieldLayerTest() {}

  virtual void SetUp() {

  }

  virtual ~MultiStageMeanfieldLayerTest() {

  }
};
TYPED_TEST_CASE(MultiStageMeanfieldLayerTest, TestDtypesAndDevices);
TYPED_TEST(MultiStageMeanfieldLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  const int n = 5, c = 3, H = 5, W = 5;

  if (sizeof(Dtype) != sizeof(float))
    return;

  Blob<Dtype> unary_terms_blob(n, c, H, W);
  Blob<Dtype> previous_output_blob(n, c, H, W);
  Blob<Dtype> rgb_blob(n, 3, H, W);

  caffe::FillAsLogProb(&unary_terms_blob);
  caffe::FillAsLogProb(&previous_output_blob);
  caffe::FillAsRGB(&rgb_blob);

  vector<Blob<Dtype>*> bottom_vec, top_vec;
  bottom_vec.push_back(&unary_terms_blob);
  bottom_vec.push_back(&previous_output_blob);
  bottom_vec.push_back(&rgb_blob);

  Blob<Dtype>* top_blob = new Blob<Dtype>();
  top_vec.push_back(top_blob);

  LayerParameter layer_param2;
  MultiStageMeanfieldParameter* ms_mf_param = layer_param2.mutable_multi_stage_meanfield_param();
  ms_mf_param->set_num_iterations(2);
  ms_mf_param->set_bilateral_filter_weight(1.0);
  ms_mf_param->set_spatial_filter_weight(1.0);
  ms_mf_param->set_compatibility_mode(MultiStageMeanfieldParameter_Mode_POTTS);
  ms_mf_param->set_theta_alpha(5);
  ms_mf_param->set_theta_beta(2);
  ms_mf_param->set_theta_gamma(3);

  MultiStageMeanfieldLayer<Dtype> layer2(layer_param2);
  layer2.SetUp(bottom_vec, top_vec);
  layer2.Forward(bottom_vec, top_vec);

  GradientChecker<Dtype> checker(1e-2, 1e-3);

  // Check gradients w.r.t. unary terms
  checker.CheckGradientExhaustive(&layer2, bottom_vec, top_vec, 0);

  // Check gradients w.r.t. previous outputs
  checker.CheckGradientExhaustive(&layer2, bottom_vec, top_vec, 1);
}
}  // namespace caffe
