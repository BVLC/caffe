#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/lifted_struct_similarity_softmax_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class LiftedStructSimilaritySoftmaxLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LiftedStructSimilaritySoftmaxLossLayerTest() : blob_bottom_data(new Blob<Dtype>(128, 64, 1, 1)),
        blob_bottom_y_(new Blob<Dtype>(128, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);  // distances~=1.0 to test both sides of margin
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data);
    blob_bottom_vec_.push_back(blob_bottom_data);
    for (int i = 0; i < blob_bottom_y_->count(); ++i) {
      //blob_bottom_y_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;  // 0 or 1
      blob_bottom_y_->mutable_cpu_data()[i] = i/2;
    }
    blob_bottom_vec_.push_back(blob_bottom_y_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~LiftedStructSimilaritySoftmaxLossLayerTest() {
    delete blob_bottom_data;
    delete blob_bottom_y_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_data;
  Blob<Dtype>* const blob_bottom_y_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LiftedStructSimilaritySoftmaxLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(LiftedStructSimilaritySoftmaxLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LiftedStructSimilaritySoftmaxLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // check the gradient for the first two bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
