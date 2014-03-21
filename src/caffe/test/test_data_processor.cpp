// Copyright 2014 kloudkl@github

#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_processor.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/test/test_caffe_main.hpp"

using std::string;
using std::stringstream;

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template<typename Dtype>
class CroppingDataProcessorTest : public ::testing::Test {
 protected:
  CroppingDataProcessorTest()
      : batch_size_(10),
        channels_(3),
        original_size_(128),
        crop_size_(117),
        blob_(new Blob<Dtype>()) {
  }
  virtual void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    blob_->Reshape(batch_size_, channels_, original_size_, original_size_);
    filler.Fill(blob_);
  }

  virtual ~CroppingDataProcessorTest() {
  }

  Blob<Dtype>* blob_;
  uint32_t batch_size_;
  uint32_t channels_;
  uint32_t original_size_;
  uint32_t crop_size_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(CroppingDataProcessorTest, Dtypes);

TYPED_TEST(CroppingDataProcessorTest, TestProcess){
  DataProcessorParameter processor_param;
  processor_param.mutable_cropping_param()->set_crop_size(this->crop_size_);

  CroppingDataProcessor<TypeParam> processor(processor_param);
  EXPECT_EQ(this->crop_size_, processor.crop_size());

  shared_ptr<Blob<TypeParam> > input_blob(this->blob_);
  const TypeParam* data = this->blob_->cpu_data();
  Caffe::Brew modes[] = {Caffe::CPU, Caffe::GPU};
  Caffe::Phase phases[] = {Caffe::TRAIN, Caffe::TEST};
  for (int i = 0; i < 2; ++i) {
    Caffe::set_mode(modes[i]);
    for (int j = 0; j < 2; ++j) {
      Caffe::set_phase(phases[j]);
      shared_ptr<Blob<TypeParam> > blob(new Blob<TypeParam>());
      processor.Process(input_blob, blob);
      EXPECT_EQ(this->batch_size_, blob->num());
      EXPECT_EQ(this->channels_, blob->channels());
      EXPECT_EQ(this->crop_size_, blob->height());
      EXPECT_EQ(this->crop_size_, blob->width());

      const TypeParam* output_data = blob->cpu_data();
      const uint32_t height_offset = processor.height_offset();
      const uint32_t width_offset = processor.width_offset();
      for (int n = 0; n < blob->num(); ++n) {
        for (int c = 0; c < blob->channels(); ++c) {
          for (int h = 0; h < this->crop_size_; ++h) {
            for (int w = 0; w < this->crop_size_; ++w) {
              EXPECT_EQ(data[this->blob_->offset(
                      n, c, h + height_offset, w + width_offset)],
                  output_data[blob->offset(n, c, h, w)])
              << "debug: n " << n << " c " << c << " h " << h << " w " << w;
            }
          }
        }
      }
    }  // for (int j = 0; j < 2; ++j) {
  }  // for (int i = 0; i < 2; ++i) {
}

}
  // namespace caffe
