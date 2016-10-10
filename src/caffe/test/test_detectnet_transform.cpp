#ifdef USE_OPENCV

#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/detectnet_transform_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

#define BACKGROUND_VALUE 0

template <typename TypeParam>
class DetectNetTransformationLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DetectNetTransformationLayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);

    this->fillDataBlob();
    this->fillLabelBlob();
  }
  virtual ~DetectNetTransformationLayerTest() {
      delete blob_bottom_data_;
      delete blob_bottom_label_;
      delete blob_top_data_;
      delete blob_top_label_;
  }

  // Fill data blob with filler
  void fillDataBlob(Filler<Dtype>* filler = NULL) {
    blob_bottom_data_->Reshape(1, 3, 32, 32);
    if (filler != NULL) {
      filler->Fill(blob_bottom_data_);
    } else {
      // default to uniform noise in [0-255]
      FillerParameter filler_param;
      filler_param.set_min(0);
      filler_param.set_max(255);
      UniformFiller<Dtype> default_filler(filler_param);
      default_filler.Fill(blob_bottom_data_);
    }
  }

  // Set label to detect one object
  void fillLabelBlob() {
    blob_bottom_label_->Reshape(1, 1, 2, 16);
    Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
    int d = blob_bottom_label_->shape()[3];
    label_data[0*d+0] = 1;  // num rows
    label_data[0*d+1] = blob_bottom_label_->shape()[3];  // num cols

    label_data[1*d+0] = 0;  // bbox topleft x
    label_data[1*d+1] = 0;  // bbox topleft y
    label_data[1*d+2] = blob_bottom_data_->width()/2;  // bbox width
    label_data[1*d+3] = blob_bottom_data_->height()/2;  // bbox height
    label_data[1*d+4] = 0;  // alpha angle
    label_data[1*d+5] = 1;  // class number
    label_data[1*d+6] = 0;  // bbox.scenario()
    label_data[1*d+7] = 0;  // Y axis rotation
    label_data[1*d+8] = 0;  // truncation
    label_data[1*d+9] = 0;  // occlusion
    label_data[1*d+10] = 0;  // object length
    label_data[1*d+11] = 0;  // object width
    label_data[1*d+12] = 0;  // object height
    label_data[1*d+13] = 0;  // location_x
    label_data[1*d+14] = 0;  // location_y
    label_data[1*d+15] = 0;  // location_z
  }

  // returns a layer parameter with all augmentations turned off
  LayerParameter layerParamNoAug() {
    LayerParameter layer_param;

    DetectNetGroundTruthParameter* groundtruth_param =
        layer_param.mutable_detectnet_groundtruth_param();
    // set output size
    groundtruth_param->set_image_size_x(this->blob_bottom_data_->width());
    groundtruth_param->set_image_size_y(this->blob_bottom_data_->height());

    DetectNetAugmentationParameter* augmentation_param =
        layer_param.mutable_detectnet_augmentation_param();
    // Turn off all augmentations
    augmentation_param->set_hue_rotation_prob(0);
    augmentation_param->set_desaturation_prob(0);
    augmentation_param->set_flip_prob(0);
    augmentation_param->set_scale_prob(0);
    augmentation_param->set_rotation_prob(0);
    augmentation_param->set_crop_prob(0);

    return layer_param;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DetectNetTransformationLayerTest, TestDtypesAndDevices);

TYPED_TEST(DetectNetTransformationLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param = this->layerParamNoAug();
  const DetectNetGroundTruthParameter groundtruth_param =
    layer_param.detectnet_groundtruth_param();
  DetectNetTransformationLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->shape(), this->blob_bottom_data_->shape());
  EXPECT_EQ(this->blob_top_label_->shape()[0],
      this->blob_bottom_label_->shape()[0]);
  EXPECT_GE(this->blob_top_label_->shape()[1], 9);
  EXPECT_EQ(this->blob_top_label_->shape()[2],
      this->blob_bottom_data_->shape()[2]/groundtruth_param.stride());
  EXPECT_EQ(this->blob_top_label_->shape()[3],
      this->blob_bottom_data_->shape()[3]/groundtruth_param.stride());
}

TYPED_TEST(DetectNetTransformationLayerTest, TestNoAugmentation) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param = this->layerParamNoAug();
  DetectNetTransformationLayer<Dtype> layer(layer_param);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check that data is unchanged
  for (int i = 0; i < this->blob_bottom_data_->count(); i++) {
    EXPECT_FLOAT_EQ(this->blob_top_data_->cpu_data()[i],
        this->blob_bottom_data_->cpu_data()[i]);
  }
}

TYPED_TEST(DetectNetTransformationLayerTest, TestAllAugmentation) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param = this->layerParamNoAug();
  DetectNetAugmentationParameter* augmentation_param =
      layer_param.mutable_detectnet_augmentation_param();
  // Turn off all augmentations
  augmentation_param->set_hue_rotation_prob(1);
  augmentation_param->set_desaturation_prob(1);
  augmentation_param->set_flip_prob(1);
  augmentation_param->set_scale_prob(1);
  augmentation_param->set_rotation_prob(1);
  augmentation_param->set_crop_prob(1);
  DetectNetTransformationLayer<Dtype> layer(layer_param);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // nothing to test
}


TYPED_TEST(DetectNetTransformationLayerTest, TestDesaturation) {
  typedef typename TypeParam::Dtype Dtype;
  // make sure we don't get unlucky with a random saturation value of 0
  Caffe::set_random_seed(1234);
  LayerParameter layer_param = this->layerParamNoAug();
  DetectNetAugmentationParameter* augmentation_param =
      layer_param.mutable_detectnet_augmentation_param();
  // turn on desaturation
  augmentation_param->set_desaturation_prob(1);
  augmentation_param->set_desaturation_max(1);
  DetectNetTransformationLayer<Dtype> layer(layer_param);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  if (this->blob_bottom_data_->channels() == 3) {
    // Check that all pixels values are higher in value
    const int ns = this->blob_bottom_data_->num();
    const int cs = this->blob_bottom_data_->channels();
    const int hs = this->blob_bottom_data_->height();
    const int ws = this->blob_bottom_data_->width();
    const Dtype* bottom_data = this->blob_bottom_data_->cpu_data();
    const Dtype* top_data = this->blob_top_data_->cpu_data();
    const Dtype eps = 1e-3;
    for (int n = 0; n < ns; n++) {
      for (int h = 0; h < hs; h++) {
        for (int w = 0; w < ws; w++) {
          const Dtype r1 = bottom_data[n*cs*hs*ws + 0*hs*ws + h*ws + w];
          const Dtype g1 = bottom_data[n*cs*hs*ws + 1*hs*ws + h*ws + w];
          const Dtype b1 = bottom_data[n*cs*hs*ws + 2*hs*ws + h*ws + w];
          const Dtype r2 = top_data[n*cs*hs*ws + 0*hs*ws + h*ws + w];
          const Dtype g2 = top_data[n*cs*hs*ws + 1*hs*ws + h*ws + w];
          const Dtype b2 = top_data[n*cs*hs*ws + 2*hs*ws + h*ws + w];
          EXPECT_GE(r2 + eps, r1);
          EXPECT_GE(g2 + eps, g1);
          EXPECT_GE(b2 + eps, b1);
        }
      }
    }
  }
}

TYPED_TEST(DetectNetTransformationLayerTest, TestHueRotation) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param = this->layerParamNoAug();
  DetectNetAugmentationParameter* augmentation_param =
      layer_param.mutable_detectnet_augmentation_param();
  // turn on hue rotation
  augmentation_param->set_hue_rotation_prob(1);
  DetectNetTransformationLayer<Dtype> layer(layer_param);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // nothing to test
}

TYPED_TEST(DetectNetTransformationLayerTest, TestFlip) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param = this->layerParamNoAug();
  DetectNetAugmentationParameter* augmentation_param =
      layer_param.mutable_detectnet_augmentation_param();
  // turn on flipping
  augmentation_param->set_flip_prob(1);
  DetectNetTransformationLayer<Dtype> layer(layer_param);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check that data is flipped
  const int ns = this->blob_bottom_data_->num();
  const int cs = this->blob_bottom_data_->channels();
  const int hs = this->blob_bottom_data_->height();
  const int ws = this->blob_bottom_data_->width();
  for (int n = 0; n < ns; n++) {
    for (int c = 0; c < cs; c++) {
      for (int h = 0; h < hs; h++) {
        for (int w = 0; w < ws; w++) {
          EXPECT_FLOAT_EQ(
              this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws + h*ws + w],
              // NOLINT_NEXT_LINE(whitespace/line_length)
              this->blob_bottom_data_->cpu_data()[n*cs*hs*ws + c*hs*ws + h*ws + (ws-w-1)]);
        }
      }
    }
  }
}

TYPED_TEST(DetectNetTransformationLayerTest, TestScaleDown) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param = this->layerParamNoAug();
  DetectNetAugmentationParameter* augmentation_param =
      layer_param.mutable_detectnet_augmentation_param();
  // turn on scaling
  augmentation_param->set_scale_prob(1);
  augmentation_param->set_scale_min(0.5);
  augmentation_param->set_scale_max(0.5);
  DetectNetTransformationLayer<Dtype> layer(layer_param);
  // fill with a constant
  FillerParameter filler_param;
  const Dtype DATA_VALUE = 10;
  filler_param.set_value(DATA_VALUE);
  ConstantFiller<Dtype> filler(filler_param);
  this->fillDataBlob(&filler);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const int ns = this->blob_bottom_data_->num();
  const int cs = this->blob_bottom_data_->channels();
  const int hs = this->blob_bottom_data_->height();
  const int ws = this->blob_bottom_data_->width();
  for (int n = 0; n < ns; n++) {
    for (int c = 0; c < cs; c++) {
      // corners should be background
      EXPECT_FLOAT_EQ(BACKGROUND_VALUE,
          this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws]);
      EXPECT_FLOAT_EQ(BACKGROUND_VALUE,
          this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws + ws-1]);
      EXPECT_FLOAT_EQ(BACKGROUND_VALUE,
          this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws + (hs-1)*ws]);
      EXPECT_FLOAT_EQ(BACKGROUND_VALUE,
          // NOLINT_NEXT_LINE(whitespace/line_length)
          this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws + (hs-1)*ws + ws-1]);
      // center should be data
      EXPECT_FLOAT_EQ(DATA_VALUE,
          // NOLINT_NEXT_LINE(whitespace/line_length)
          this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws + (hs/2)*ws + ws/2]);
    }
  }
}

TYPED_TEST(DetectNetTransformationLayerTest, TestScaleUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param = this->layerParamNoAug();
  DetectNetAugmentationParameter* augmentation_param =
      layer_param.mutable_detectnet_augmentation_param();
  // turn on scaling
  augmentation_param->set_scale_prob(1);
  augmentation_param->set_scale_min(2);
  augmentation_param->set_scale_max(2);
  DetectNetTransformationLayer<Dtype> layer(layer_param);
  // fill with a constant
  FillerParameter filler_param;
  const Dtype DATA_VALUE = 10;
  filler_param.set_value(DATA_VALUE);
  ConstantFiller<Dtype> filler(filler_param);
  this->fillDataBlob(&filler);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // check results
  const int ns = this->blob_bottom_data_->num();
  const int cs = this->blob_bottom_data_->channels();
  const int hs = this->blob_bottom_data_->height();
  const int ws = this->blob_bottom_data_->width();
  for (int n = 0; n < ns; n++) {
    for (int c = 0; c < cs; c++) {
      // corners should be data
      EXPECT_FLOAT_EQ(DATA_VALUE,
          this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws]);
      EXPECT_FLOAT_EQ(DATA_VALUE,
          this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws + ws-1]);
      EXPECT_FLOAT_EQ(DATA_VALUE,
          this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws + (hs-1)*ws]);
      EXPECT_FLOAT_EQ(DATA_VALUE,
          // NOLINT_NEXT_LINE(whitespace/line_length)
          this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws + (hs-1)*ws + ws-1]);
      // center should be data
      EXPECT_FLOAT_EQ(DATA_VALUE,
          // NOLINT_NEXT_LINE(whitespace/line_length)
          this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws + (hs/2)*ws + ws/2]);
    }
  }
}

TYPED_TEST(DetectNetTransformationLayerTest, TestRotation) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param = this->layerParamNoAug();
  DetectNetAugmentationParameter* augmentation_param =
      layer_param.mutable_detectnet_augmentation_param();
  // turn on rotation
  augmentation_param->set_rotation_prob(1);
  augmentation_param->set_max_rotate_degree(80);
  // downscale by a bit
  augmentation_param->set_scale_prob(1);
  augmentation_param->set_scale_min(0.85);
  augmentation_param->set_scale_max(0.85);
  DetectNetTransformationLayer<Dtype> layer(layer_param);
  // fill with a constant
  FillerParameter filler_param;
  const Dtype DATA_VALUE = 10;
  filler_param.set_value(DATA_VALUE);
  ConstantFiller<Dtype> filler(filler_param);
  this->fillDataBlob(&filler);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // check results
  const int ns = this->blob_bottom_data_->num();
  const int cs = this->blob_bottom_data_->channels();
  const int hs = this->blob_bottom_data_->height();
  const int ws = this->blob_bottom_data_->width();
  for (int n = 0; n < ns; n++) {
    for (int c = 0; c < cs; c++) {
      // corners should be background
      EXPECT_FLOAT_EQ(BACKGROUND_VALUE,
          this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws]);
      EXPECT_FLOAT_EQ(BACKGROUND_VALUE,
          this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws + ws-1]);
      EXPECT_FLOAT_EQ(BACKGROUND_VALUE,
          this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws + (hs-1)*ws]);
      EXPECT_FLOAT_EQ(BACKGROUND_VALUE,
          // NOLINT_NEXT_LINE(whitespace/line_length)
          this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws + (hs-1)*ws + ws-1]);
      // center should be data
      EXPECT_FLOAT_EQ(DATA_VALUE,
          // NOLINT_NEXT_LINE(whitespace/line_length)
          this->blob_top_data_->cpu_data()[n*cs*hs*ws + c*hs*ws + (hs/2)*ws + ws/2]);
    }
  }
}

TYPED_TEST(DetectNetTransformationLayerTest, TestCrop) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param = this->layerParamNoAug();
  DetectNetAugmentationParameter* augmentation_param =
      layer_param.mutable_detectnet_augmentation_param();
  // turn on cropping
  augmentation_param->set_crop_prob(1);
  augmentation_param->set_shift_x(this->blob_bottom_data_->width());
  augmentation_param->set_shift_y(this->blob_bottom_data_->height());
  DetectNetTransformationLayer<Dtype> layer(layer_param);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // nothing to test
}


}  // namespace caffe

#endif  // USE_OPENCV
