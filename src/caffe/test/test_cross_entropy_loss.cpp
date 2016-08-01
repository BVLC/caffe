#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/cross_entropy_loss.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_sigmoid_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CrossEntropyLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CrossEntropyLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_targets_(new Blob<Dtype>(10, 5, 1, 1)),
	blob_top_sig_(new Blob<Dtype>()),
        blob_top_loss_(new Blob<Dtype>()) {
    // Fill the data vector
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    data_filler.Fill(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    // Set bottom sig vector
    blob_top_vec_sig_.push_back(blob_top_sig_);
    blob_tmp_vec_.push_back(blob_top_sig_);
    
    // Fill the targets vector
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(0);
    targets_filler_param.set_max(1);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    targets_filler.Fill(blob_bottom_targets_);
    blob_tmp_vec_.push_back(blob_bottom_targets_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~CrossEntropyLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_targets_;
    delete blob_top_sig_;
    delete blob_top_loss_;
  }

  Dtype CrossEntropyReference(const int count, const int num,
				  const Dtype* input,
				  const Dtype* target) {
    Dtype loss = 0;
    for (int i = 0; i < count; ++i) {
      const Dtype prediction = 1 / (1 + exp(-input[i]));
      EXPECT_LE(prediction, 1);
      EXPECT_GE(prediction, 0);
      EXPECT_LE(target[i], 1);
      EXPECT_GE(target[i], 0);
      loss -= target[i] * log(prediction + (target[i] == Dtype(0)));
      loss -= (1 - target[i]) * log(1 - prediction + (target[i] == Dtype(1)));
    }
    return loss / num;
  }

  void TestForward() {
    LayerParameter layer_param_sigmoid;
    LayerParameter layer_param;
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(0.0);
    targets_filler_param.set_max(1.0);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    Dtype eps = 2e-2;
    for (int i = 0; i < 100; ++i) {
      // Fill the data vector
      data_filler.Fill(this->blob_bottom_data_);
      // Fill the targets vector
      targets_filler.Fill(this->blob_bottom_targets_);

      // sigmoid layer
      SigmoidLayer<Dtype> siglayer(layer_param_sigmoid);
      siglayer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_sig_);
      siglayer.Forward(this->blob_bottom_vec_, this->blob_top_vec_sig_);
      
      // cross entropy layer
      CrossEntropyLossLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_tmp_vec_, this->blob_top_vec_);
      Dtype layer_loss =
          layer.Forward(this->blob_tmp_vec_, this->blob_top_vec_);
      const int count = this->blob_bottom_data_->count();
      const int num = this->blob_bottom_data_->num();
      const Dtype* blob_bottom_data = this->blob_bottom_data_->cpu_data();
      const Dtype* blob_bottom_targets =
          this->blob_bottom_targets_->cpu_data();
      Dtype reference_loss = kLossWeight * CrossEntropyReference(
          count, num, blob_bottom_data, blob_bottom_targets);
      EXPECT_NEAR(reference_loss, layer_loss, eps) << "debug: trial #" << i;
    }
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_targets_;
  Blob<Dtype>* const blob_top_sig_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_, blob_tmp_vec_;
  vector<Blob<Dtype>*> blob_top_vec_,blob_top_vec_sig_;
};

TYPED_TEST_CASE(CrossEntropyLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(CrossEntropyLossLayerTest, TestCrossEntropy) {
  this->TestForward();
}

}  // namespace caffe
