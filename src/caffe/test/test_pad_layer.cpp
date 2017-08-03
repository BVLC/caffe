#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/pad_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class PadLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PadLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 4, 5, 4)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~PadLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};


TYPED_TEST_CASE(PadLayerTest, TestDtypesAndDevices);

TYPED_TEST(PadLayerTest, SetupShapeDefault) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    EXPECT_EQ(this->blob_bottom_->shape(i), this->blob_top_->shape(i));
  }
}

TYPED_TEST(PadLayerTest, SetupShapePad1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Pad width 1
  layer_param.mutable_pad_param()->set_pad(1);
  PadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    if (i < 2) {
      EXPECT_EQ(this->blob_bottom_->shape(i), this->blob_top_->shape(i));
    } else {
      EXPECT_EQ(this->blob_bottom_->shape(i)+2, this->blob_top_->shape(i));
    }
  }
}

TYPED_TEST(PadLayerTest, SetupShapePad2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Pad width 2
  layer_param.mutable_pad_param()->set_pad(2);
  PadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    if (i < 2) {
      EXPECT_EQ(this->blob_bottom_->shape(i), this->blob_top_->shape(i));
    } else {
      EXPECT_EQ(this->blob_bottom_->shape(i)+4, this->blob_top_->shape(i));
    }
  }
}

TYPED_TEST(PadLayerTest, ForwardDefault) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Pad width of 0
  // layer_param.mutable_pad_param()->set_pad(0);
  PadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
	  // If one fails, don't continue with a bazillion messages
	  ASSERT_EQ(this->blob_top_->data_at(n, c, h, w),
		    this->blob_bottom_->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(PadLayerTest, ForwardZeroPad1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Pad width of 1
  layer_param.mutable_pad_param()->set_padtype(PadParameter::ZERO);
  layer_param.mutable_pad_param()->set_pad(1);
  PadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const int
    bredge = this->blob_bottom_->width()-1,
    tredge = this->blob_top_->width()-1,
    bbedge = this->blob_bottom_->height()-1,
    tbedge = this->blob_top_->height()-1;
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h <= bbedge; ++h) {
        for (int w = 0; w <= bredge; ++w) {
	  // If one fails, don't continue with a bazillion messages
	  ASSERT_EQ(this->blob_bottom_->data_at(n, c, h, w),
		    this->blob_top_->data_at(n, c, h+1, w+1));
        } // w
	// Horizontal padding
	ASSERT_EQ(static_cast<Dtype>(0), this->blob_top_->data_at(n, c, h+1, 0));
	ASSERT_EQ(static_cast<Dtype>(0), this->blob_top_->data_at(n, c, h+1, tredge));
      } // h
      // Vertical padding
      for (int w = 0; w < this->blob_top_->width(); ++w) {
	ASSERT_EQ(static_cast<Dtype>(0), this->blob_top_->data_at(n, c, 0, w));
	ASSERT_EQ(static_cast<Dtype>(0), this->blob_top_->data_at(n, c, tbedge, w));
      } // w
    } // c
  } // n
}

TYPED_TEST(PadLayerTest, ForwardZeroPad2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Pad width of 2, replicate
  layer_param.mutable_pad_param()->set_padtype(PadParameter::ZERO);
  layer_param.mutable_pad_param()->set_pad(2);
  PadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const int
    bredge = this->blob_bottom_->width()-1,
    tredge = this->blob_top_->width()-1,
    bbedge = this->blob_bottom_->height()-1,
    tbedge = this->blob_top_->height()-1;
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h <= bbedge; ++h) {
        for (int w = 0; w <= bredge; ++w) {
	  // If one fails, don't continue with a bazillion messages
	  ASSERT_EQ(this->blob_bottom_->data_at(n, c, h, w),
		    this->blob_top_->data_at(n, c, h+2, w+2));
	} // w
	// Horizontal padding
	ASSERT_EQ(static_cast<Dtype>(0), this->blob_top_->data_at(n, c, h+2, 0));
	ASSERT_EQ(static_cast<Dtype>(0), this->blob_top_->data_at(n, c, h+2, 1));
	ASSERT_EQ(static_cast<Dtype>(0), this->blob_top_->data_at(n, c, h+2, tredge));
	ASSERT_EQ(static_cast<Dtype>(0), this->blob_top_->data_at(n, c, h+2, tredge-1));
      } // h
      // Vertical padding
      for (int w = 0; w < this->blob_top_->width(); ++w) {
	ASSERT_EQ(static_cast<Dtype>(0), this->blob_top_->data_at(n, c, 0, w));
	ASSERT_EQ(static_cast<Dtype>(0), this->blob_top_->data_at(n, c, 1, w));
	ASSERT_EQ(static_cast<Dtype>(0), this->blob_top_->data_at(n, c, tbedge, w));
	ASSERT_EQ(static_cast<Dtype>(0), this->blob_top_->data_at(n, c, tbedge-1, w));
      } // w
    } // c
  } // n
}

TYPED_TEST(PadLayerTest, ForwardReplPad1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Pad width of 1
  layer_param.mutable_pad_param()->set_padtype(PadParameter::REPLICATE);
  layer_param.mutable_pad_param()->set_pad(1);
  PadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
	  // If one fails, don't continue with a bazillion messages
	  ASSERT_EQ(this->blob_top_->data_at(n, c, h+1, w+1),
		    this->blob_bottom_->data_at(n, c, h, w));
        } // w
	// Horizontal padding
	ASSERT_EQ(this->blob_top_->data_at(n, c, h+1, 0),
		  this->blob_bottom_->data_at(n, c, h, 0));
	ASSERT_EQ(this->blob_top_->data_at(n, c, h+1, this->blob_top_->width()-1),
		  this->blob_bottom_->data_at(n, c, h, this->blob_bottom_->width()-1));
      } // h
      // Vertical padding
      for (int w = 0; w < this->blob_top_->width(); ++w) {
	ASSERT_EQ(this->blob_top_->data_at(n, c, 0, w),
		  this->blob_top_->data_at(n, c, 1, w));
	ASSERT_EQ(this->blob_top_->data_at(n, c, this->blob_top_->height()-1, w),
		  this->blob_top_->data_at(n, c, this->blob_top_->height()-2, w));
      } // w
    } // c
  } // n
}

TYPED_TEST(PadLayerTest, ForwardReplPad2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Pad width of 2, replicate
  layer_param.mutable_pad_param()->set_padtype(PadParameter::REPLICATE);
  layer_param.mutable_pad_param()->set_pad(2);
  PadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const int
    bredge = this->blob_bottom_->width()-1,
    tredge = this->blob_top_->width()-1,
    bbedge = this->blob_bottom_->height()-1,
    tbedge = this->blob_top_->height()-1;
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h <= bbedge; ++h) {
        for (int w = 0; w <= bredge; ++w) {
	  // If one fails, don't continue with a bazillion messages
	  ASSERT_EQ(this->blob_top_->data_at(n, c, h+2, w+2),
		    this->blob_bottom_->data_at(n, c, h, w));
	} // w
	// Horizontal padding
	ASSERT_EQ(this->blob_top_->data_at(n, c, h+2, 0),
		  this->blob_bottom_->data_at(n, c, h, 0));
	ASSERT_EQ(this->blob_top_->data_at(n, c, h+2, 1),
		  this->blob_bottom_->data_at(n, c, h, 0));
	ASSERT_EQ(this->blob_top_->data_at(n, c, h+2, tredge),
		  this->blob_bottom_->data_at(n, c, h, bredge));
	ASSERT_EQ(this->blob_top_->data_at(n, c, h+2, tredge-1),
		  this->blob_bottom_->data_at(n, c, h, bredge));
      } // h
      // Vertical padding
      for (int w = 0; w < this->blob_top_->width(); ++w) {
	const int
	  wb = std::min(bredge, std::max(0, w-2));

	ASSERT_EQ(this->blob_top_->data_at(n, c, 0, w),
		  this->blob_bottom_->data_at(n, c, 0, wb));
	ASSERT_EQ(this->blob_top_->data_at(n, c, 1, w),
		  this->blob_bottom_->data_at(n, c, 0, wb));
	ASSERT_EQ(this->blob_top_->data_at(n, c, tbedge, w),
		  this->blob_bottom_->data_at(n, c, bbedge, wb));
	ASSERT_EQ(this->blob_top_->data_at(n, c, tbedge-1, w),
		  this->blob_bottom_->data_at(n, c, bbedge, wb));
      } // w
    } // c
  } // n
}

TYPED_TEST(PadLayerTest, GradientZeroPad1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Pad width of 1
  layer_param.mutable_pad_param()->set_padtype(PadParameter::ZERO);
  layer_param.mutable_pad_param()->set_pad(1);
  PadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(PadLayerTest, GradientReplPad1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Pad width of 1
  layer_param.mutable_pad_param()->set_padtype(PadParameter::REPLICATE);
  layer_param.mutable_pad_param()->set_pad(1);
  PadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(PadLayerTest, GradientZeroPad2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Pad width of 2
  layer_param.mutable_pad_param()->set_padtype(PadParameter::ZERO);
  layer_param.mutable_pad_param()->set_pad(2);
  PadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(PadLayerTest, GradientReplPad2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Pad width of 2
  layer_param.mutable_pad_param()->set_padtype(PadParameter::REPLICATE);
  layer_param.mutable_pad_param()->set_pad(2);
  PadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
