#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
#include "caffe/loss_layers.hpp"
=======
=======
>>>>>>> BVLC/device-abstraction
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod/caffe-merge
=======
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod/caffe-merge
=======
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/loss_layers.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/loss_layers.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/loss_layers.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/loss_layers.hpp"
>>>>>>> master
=======
#include "caffe/loss_layers.hpp"
>>>>>>> caffe
=======
#include "caffe/loss_layers.hpp"
>>>>>>> master
=======
#include "caffe/loss_layers.hpp"
>>>>>>> master
=======
#include "caffe/loss_layers.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/loss_layers.hpp"
>>>>>>> master
=======
#include "caffe/loss_layers.hpp"
>>>>>>> master
=======
#include "caffe/loss_layers.hpp"
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
=======
#include "caffe/loss_layers.hpp"
>>>>>>> BVLC/master
>>>>>>> device-abstraction
=======
#include "caffe/layers/hinge_loss_layer.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/layers/hinge_loss_layer.hpp"
>>>>>>> BVLC/master
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
=======
#include "caffe/loss_layers.hpp"
>>>>>>> BVLC/master
>>>>>>> device-abstraction

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class HingeLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  HingeLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~HingeLossLayerTest() {
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

TYPED_TEST_CASE(HingeLossLayerTest, TestDtypesAndDevices);


TYPED_TEST(HingeLossLayerTest, TestGradientL1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  HingeLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 2e-3, 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(HingeLossLayerTest, TestGradientL2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Set norm to L2
  HingeLossParameter* hinge_loss_param = layer_param.mutable_hinge_loss_param();
  hinge_loss_param->set_norm(HingeLossParameter_Norm_L2);
  HingeLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
