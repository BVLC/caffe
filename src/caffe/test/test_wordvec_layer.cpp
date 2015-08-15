#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using std::min;
using std::max;

#define BATCH_SIZE 2
#define VOCAB_SIZE 10
#define DIMENSION 15
#define SENTENCE_LENGTH 1

namespace caffe {

template <typename TypeParam>
class WordvecLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  WordvecLayerTest()
      : epsilon_(Dtype(1e-5)),
        blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1601);
    blob_bottom_->Reshape(BATCH_SIZE, SENTENCE_LENGTH, 1, 1);
    // fill the values
    Dtype * bottom_data = this->blob_bottom_->mutable_cpu_data();
    for (int n = 0; n < BATCH_SIZE; ++n) {
      for (int i = 0; i < SENTENCE_LENGTH; ++i) {
        bottom_data[i + n * SENTENCE_LENGTH] = caffe_rng_rand() % VOCAB_SIZE;
      }
    }
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~WordvecLayerTest() { delete blob_bottom_; delete blob_top_; }
  void ReferenceWordvecForward(const Blob<Dtype>& blob_bottom,
      const LayerParameter& layer_param, Blob<Dtype>* blob_top);

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename TypeParam>
void WordvecLayerTest<TypeParam>::ReferenceWordvecForward(
    const Blob<Dtype>& blob_bottom, const LayerParameter& layer_param,
    Blob<Dtype>* blob_top) {
  typedef typename TypeParam::Dtype Dtype;
  blob_top->Reshape(blob_bottom.num(), blob_bottom.channels(),
      blob_bottom.height(), blob_bottom.width());
  Dtype* top_data = blob_top->mutable_cpu_data();
  WordvecParameter wordvec_param = layer_param.wordvec_param();
}

TYPED_TEST_CASE(WordvecLayerTest, TestDtypesAndDevices);

TYPED_TEST(WordvecLayerTest, TestSetupAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WordvecParameter* wordvec_param = layer_param.mutable_wordvec_param();
  wordvec_param->set_vocab_size(VOCAB_SIZE);
  wordvec_param->set_dimension(DIMENSION);
  wordvec_param->mutable_weight_filler()->set_type("xavier");
  WordvecLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), BATCH_SIZE);
  EXPECT_EQ(this->blob_top_->channels(), DIMENSION);
  EXPECT_EQ(this->blob_top_->height(), SENTENCE_LENGTH);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(WordvecLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WordvecParameter* wordvec_param = layer_param.mutable_wordvec_param();
  wordvec_param->set_vocab_size(VOCAB_SIZE);
  wordvec_param->set_dimension(DIMENSION);
  wordvec_param->mutable_weight_filler()->set_type("xavier");
  WordvecLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 1e-2, 1601);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                 this->blob_top_vec_, -2);
}

}  // namespace caffe
