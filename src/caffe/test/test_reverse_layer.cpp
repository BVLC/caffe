#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/reverse_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class ReverseLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ReverseLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    blob_bottom_vec_.push_back(blob_bottom_);
  }


  virtual ~ReverseLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  void TestForwardAxis(
          int axis,
          int s0,
          int s1,
          int s2,
          int s3,
          const Dtype data_in[],
          const Dtype data_expected[]) {
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    LayerParameter layer_param;
    ReverseParameter* reverse_param =
        layer_param.mutable_reverse_param();

    reverse_param->set_axis(axis);

    shared_ptr<ReverseLayer<Dtype> > layer(
          new ReverseLayer<Dtype>(layer_param));

    // create dummy data and diff
    blob_bottom_->Reshape(5, 2, 1, 3);
    blob_top_->ReshapeLike(*blob_bottom_);

    // copy input data
    caffe_copy(blob_bottom_->count(), data_in,
               blob_bottom_->mutable_cpu_data());

    // Forward data
    layer->Forward(blob_bottom_vec_, blob_top_vec_);

    // Output of top must match the expected data
    EXPECT_EQ(blob_bottom_->count(), blob_top_->count());

    for (int i = 0; i < blob_top_->count(); ++i) {
      EXPECT_FLOAT_EQ(data_expected[i], blob_top_->cpu_data()[i]);
    }
  }

  void TestBackwardAxis(
          int axis,
          int s0,
          int s1,
          int s2,
          int s3,
          const Dtype diff_in[],
          const Dtype diff_expected[]) {
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    LayerParameter layer_param;
    ReverseParameter* reverse_param =
        layer_param.mutable_reverse_param();
    reverse_param->set_axis(axis);

    shared_ptr<ReverseLayer<Dtype> > layer(
          new ReverseLayer<Dtype>(layer_param));

    // create dummy data and diff
    blob_bottom_->Reshape(5, 2, 1, 3);
    blob_top_->ReshapeLike(*blob_bottom_);

    // copy input diff
    caffe_copy(blob_top_->count(), diff_in, blob_top_->mutable_cpu_diff());

    // Backward diff
    layer->Backward(blob_top_vec_, vector<bool>(1, true), blob_bottom_vec_);

    // Output of top must match the expected data
    EXPECT_EQ(blob_bottom_->count(), blob_top_->count());

    for (int i = 0; i < blob_top_->count(); ++i) {
      EXPECT_FLOAT_EQ(diff_expected[i], blob_bottom_->cpu_diff()[i]);
    }
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ReverseLayerTest, TestDtypesAndDevices);

TYPED_TEST(ReverseLayerTest, TestForwardAxisZero) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype data_in[5 * 2 * 1 * 3] = {
    1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30
  };

  // first axis must be inverted
  const Dtype data_expected[5 * 2 * 1 * 3] = {
    25, 26, 27, 28, 29, 30,
    19, 20, 21, 22, 23, 24,
    13, 14, 15, 16, 17, 18,
    7, 8, 9, 10, 11, 12,
    1, 2, 3, 4, 5, 6
  };


  this->TestForwardAxis(0, 5, 2, 1, 3, data_in, data_expected);
}

TYPED_TEST(ReverseLayerTest, TestBackwardAxisZero) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype diff_in[5 * 2 * 1 * 3] = {
    100, 101, 102, 103, 104, 105,
    106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 122, 123,
    124, 125, 126, 127, 128, 129
  };

  // first axis must be inverted
  const Dtype diff_expected[5 * 2 * 1 * 3] = {
    124, 125, 126, 127, 128, 129,
    118, 119, 120, 121, 122, 123,
    112, 113, 114, 115, 116, 117,
    106, 107, 108, 109, 110, 111,
    100, 101, 102, 103, 104, 105
  };

  this->TestBackwardAxis(0, 5, 2, 1, 3, diff_in, diff_expected);
}

TYPED_TEST(ReverseLayerTest, TestForwardAxisOne) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype data_in[5 * 2 * 1 * 3] = {
    1, 2, 3,
    4, 5, 6,

    7, 8, 9,
    10, 11, 12,

    13, 14, 15,
    16, 17, 18,

    19, 20, 21,
    22, 23, 24,

    25, 26, 27,
    28, 29, 30
  };

  // second axis must be inverted
  const Dtype data_expected[5 * 2 * 1 * 3] = {
    4, 5, 6,
    1, 2, 3,

    10, 11, 12,
    7, 8, 9,

    16, 17, 18,
    13, 14, 15,

    22, 23, 24,
    19, 20, 21,

    28, 29, 30,
    25, 26, 27
  };

  this->TestForwardAxis(1, 5, 2, 1, 3, data_in, data_expected);
}

TYPED_TEST(ReverseLayerTest, TestBackwardAxisOne) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype diff_in[5 * 2 * 1 * 3] = {
    100, 101, 102,
    103, 104, 105,
    106, 107, 108,
    109, 110, 111,
    112, 113, 114,
    115, 116, 117,
    118, 119, 120,
    121, 122, 123,
    124, 125, 126,
    127, 128, 129
  };

  // first axis must be inverted
  const Dtype diff_expected[5 * 2 * 1 * 3] = {
    103, 104, 105,
    100, 101, 102,
    109, 110, 111,
    106, 107, 108,
    115, 116, 117,
    112, 113, 114,
    121, 122, 123,
    118, 119, 120,
    127, 128, 129,
    124, 125, 126
  };

  this->TestBackwardAxis(1, 5, 2, 1, 3, diff_in, diff_expected);
}

TYPED_TEST(ReverseLayerTest, TestForwardAxisTwo) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype data_in[5 * 2 * 1 * 3] = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10, 11, 12,
    13, 14, 15,
    16, 17, 18,
    19, 20, 21,
    22, 23, 24,
    25, 26, 27,
    28, 29, 30
  };

  // second axis must be inverted
  const Dtype data_expected[5 * 2 * 1 * 3] = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10, 11, 12,
    13, 14, 15,
    16, 17, 18,
    19, 20, 21,
    22, 23, 24,
    25, 26, 27,
    28, 29, 30
  };

  this->TestForwardAxis(2, 5, 2, 1, 3, data_in, data_expected);
}

TYPED_TEST(ReverseLayerTest, TestBackwardAxisTwo) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype diff_in[5 * 2 * 1 * 3] = {
    100, 101, 102,
    103, 104, 105,
    106, 107, 108,
    109, 110, 111,
    112, 113, 114,
    115, 116, 117,
    118, 119, 120,
    121, 122, 123,
    124, 125, 126,
    127, 128, 129
  };

  // first axis must be inverted
  const Dtype diff_expected[5 * 2 * 1 * 3] = {
    100, 101, 102,
    103, 104, 105,
    106, 107, 108,
    109, 110, 111,
    112, 113, 114,
    115, 116, 117,
    118, 119, 120,
    121, 122, 123,
    124, 125, 126,
    127, 128, 129
  };

  this->TestBackwardAxis(2, 5, 2, 1, 3, diff_in, diff_expected);
}

TYPED_TEST(ReverseLayerTest, TestForwardAxisThree) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype data_in[5 * 2 * 1 * 3] = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10, 11, 12,
    13, 14, 15,
    16, 17, 18,
    19, 20, 21,
    22, 23, 24,
    25, 26, 27,
    28, 29, 30
  };

  // second axis must be inverted
  const Dtype data_expected[5 * 2 * 1 * 3] = {
    3, 2, 1,
    6, 5, 4,
    9, 8, 7,
    12, 11, 10,
    15, 14, 13,
    18, 17, 16,
    21, 20, 19,
    24, 23, 22,
    27, 26, 25,
    30, 29, 28
  };

  this->TestForwardAxis(3, 5, 2, 1, 3, data_in, data_expected);
}

TYPED_TEST(ReverseLayerTest, TestBackwardAxisThree) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype diff_in[5 * 2 * 1 * 3] = {
    100, 101, 102,
    103, 104, 105,
    106, 107, 108,
    109, 110, 111,
    112, 113, 114,
    115, 116, 117,
    118, 119, 120,
    121, 122, 123,
    124, 125, 126,
    127, 128, 129
  };

  // first axis must be inverted
  const Dtype diff_expected[5 * 2 * 1 * 3] = {
    102, 101, 100,
    105, 104, 103,
    108, 107, 106,
    111, 110, 109,
    114, 113, 112,
    117, 116, 115,
    120, 119, 118,
    123, 122, 121,
    126, 125, 124,
    129, 128, 127
  };

  this->TestBackwardAxis(3, 5, 2, 1, 3, diff_in, diff_expected);
}

}  // namespace caffe
