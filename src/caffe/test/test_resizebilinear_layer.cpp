#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/resizebilinear_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ResizeBilinearLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  ResizeBilinearLayerTest()
    : blob_bottom_(new Blob<Dtype>()),
      blob_top_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values;
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ResizeBilinearLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForward() {
    LayerParameter layer_param;
    ResizeBilinearParameter* rebilinear_param = layer_param.mutable_resize_bilinear_param();
    rebilinear_param->set_factor(2);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 2, 2);
    // test case 1: NCHW(2, 2, 2, 2)
    //   [1 3]
    //   [4 9]
    // test case 2: NCHW(2, 2, 3, 5)
    //   [1 2 5 2 3]
    //   [9 4 1 4 8]
    //   [1 2 4 2 3]
    for (int i = 0; i < 4 * num * channels; i += 4) {
      blob_bottom_->mutable_cpu_data()[i + 0] = 1;
      blob_bottom_->mutable_cpu_data()[i + 1] = 3;
      blob_bottom_->mutable_cpu_data()[i + 2] = 4;
      blob_bottom_->mutable_cpu_data()[i + 3] = 9;
  //  blob_bottom_->mutable_cpu_data()[i + 4] = 3;
  //  blob_bottom_->mutable_cpu_data()[i + 5] = 9;
  //  blob_bottom_->mutable_cpu_data()[i + 6] = 4;
  //  blob_bottom_->mutable_cpu_data()[i + 7] = 1;
  //  blob_bottom_->mutable_cpu_data()[i + 8] = 4;
  //  blob_bottom_->mutable_cpu_data()[i + 9] = 8;
  //  blob_bottom_->mutable_cpu_data()[i + 10] = 1;
  //  blob_bottom_->mutable_cpu_data()[i + 11] = 2;
  //  blob_bottom_->mutable_cpu_data()[i + 12] = 4;
  //  blob_bottom_->mutable_cpu_data()[i + 13] = 2;
  //  blob_bottom_->mutable_cpu_data()[i + 14] = 3;
    }
    ResizeBilinearLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 4);
    EXPECT_EQ(blob_top_->width(), 4);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // test case 1 expected output: NCHW(2, 2, 4, 4)
    //   [1.    2.      3.      3.]
    //   [2.5   4.25    6.      6.]
    //   [4.    6.5     9.      9.]
    //   [4.    6.5     9.      9.]
    // test case 2 expected output: NCHW(2, 2, 6, 10)
    //   [1.    1.5     2.      3.5     5.      3.5     2.       2.5     3.      3.   ]
    //   [5.    4.      3.      3.      3.      3.      3.       4.25    5.5     5.5  ]
    //   [9.    6.5     4.      2.5     1.      2.5     4.       6.      8.      8.   ]
    //   [5.    4.      3.      2.75    2.5     2.75    3.       4.25    5.5     5.5  ]
    //   [1.    1.5     2.      3.      4.      3.      2.       2.5     3.      3.   ]
    //   [1.    1.5     2.      3.      4.      3.      2.       2.5     3.      3.   ]
    Dtype epsilon = 1e-8;
    for (int i = 0; i < 16 * num * channels; i += 16) {
      EXPECT_NEAR(blob_top_->cpu_data()[i + 0], 1., epsilon);
      EXPECT_NEAR(blob_top_->cpu_data()[i + 1], 2., epsilon);
      EXPECT_NEAR(blob_top_->cpu_data()[i + 2], 3., epsilon);
      EXPECT_NEAR(blob_top_->cpu_data()[i + 3], 3., epsilon);
      EXPECT_NEAR(blob_top_->cpu_data()[i + 4], 2.5, epsilon);
      EXPECT_NEAR(blob_top_->cpu_data()[i + 5], 4.25, epsilon);
      EXPECT_NEAR(blob_top_->cpu_data()[i + 6], 6., epsilon);
      EXPECT_NEAR(blob_top_->cpu_data()[i + 7], 6., epsilon);
      EXPECT_NEAR(blob_top_->cpu_data()[i + 8], 4., epsilon);
      EXPECT_NEAR(blob_top_->cpu_data()[i + 9], 6.5, epsilon);
      EXPECT_NEAR(blob_top_->cpu_data()[i + 10], 9., epsilon);
      EXPECT_NEAR(blob_top_->cpu_data()[i + 11], 9., epsilon);
      EXPECT_NEAR(blob_top_->cpu_data()[i + 12], 4., epsilon);
      EXPECT_NEAR(blob_top_->cpu_data()[i + 13], 6.5, epsilon);
      EXPECT_NEAR(blob_top_->cpu_data()[i + 14], 9., epsilon);
      EXPECT_NEAR(blob_top_->cpu_data()[i + 15], 9., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 16], 3., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 17], 4.25, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 18], 5.5, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 19], 5.5, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 20], 9., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 21], 6.5, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 22], 4., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 23], 2.5, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 24], 1., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 25], 2.5, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 26], 4., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 27], 6., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 28], 8., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 29], 8., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 30], 5., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 31], 4., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 32], 3., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 33], 2.75, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 34], 2.5, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 35], 2.75, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 36], 3., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 37], 4.25, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 38], 5.5, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 39], 5.5, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 40], 1., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 41], 1.5, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 42], 2., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 43], 3., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 44], 4., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 45], 3., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 46], 2., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 47], 2.5, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 48], 3., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 49], 3., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 50], 1., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 51], 1.5, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 52], 2., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 53], 3., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 54], 4., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 55], 3., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 56], 2., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 57], 2.5, epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 58], 3., epsilon);
//    EXPECT_NEAR(blob_top_->cpu_data()[i + 59], 3., epsilon);
    }
  }
};

TYPED_TEST_CASE(ResizeBilinearLayerTest, TestDtypesAndDevices);

TYPED_TEST(ResizeBilinearLayerTest, TestResizeBilinear) {
  this->TestForward();
}

}  // namespace caffe

