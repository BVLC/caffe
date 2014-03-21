// Copyright 2014 kloudkl@github

#include <algorithm>  // std::count
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

namespace caffe {
using std::string;
using std::vector;

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template<typename Dtype>
class DataProcessorTest : public ::testing::Test {
 protected:
  DataProcessorTest()
      : filler_(filler_param_),
        batch_size_(16),
        channels_(8),
        original_size_(64),
        crop_size_(55),
        blob_(new Blob<Dtype>()) {
  }
  virtual void SetUp() {
    blob_->Reshape(batch_size_, channels_, original_size_, original_size_);
    filler_.Fill(blob_);
  }

  virtual ~DataProcessorTest() {
  }

  FillerParameter filler_param_;
  GaussianFiller<Dtype> filler_;
  Blob<Dtype>* blob_;
  uint32_t batch_size_;
  uint32_t channels_;
  uint32_t original_size_;
  uint32_t crop_size_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(DataProcessorTest, Dtypes);

TYPED_TEST(DataProcessorTest, TestCroppingDataProcessor_Process){
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
      BLOB_ALL_DIMS_LOOP_BEGIN(blob->num(), blob->channels(),  blob->height(),
                                 blob->width())
            EXPECT_EQ(data[this->blob_->offset(
                    n, c, h + height_offset, w + width_offset)],
                output_data[blob->offset(n, c, h, w)])
            << "debug: n " << n << " c " << c << " h " << h << " w " << w;
      BLOB_ALL_DIMS_LOOP_END
    }  // for (int j = 0; j < 2; ++j) {
  }  // for (int i = 0; i < 2; ++i) {
}

TYPED_TEST(DataProcessorTest, TestMirroringDataProcessor_Process){
  DataProcessorParameter processor_param;
  shared_ptr<Blob<TypeParam> > input_blob(this->blob_);
  const TypeParam* data = this->blob_->cpu_data();
  Caffe::Brew modes[] = {Caffe::CPU, Caffe::GPU};
  Caffe::Phase phases[] = {Caffe::TRAIN, Caffe::TEST};
  MirroringParameter::MirroringType types[] = {MirroringParameter::LEFT_RIGHT,
    MirroringParameter::UP_DOWN, MirroringParameter::LEFT_RIGHT_AND_UP_DOWN};
  float random_sampling_ratios[] = {0, 0.5, 1};
  int num_is_mirrored;
  vector<bool> is_mirrored;
  int batch_size = this->batch_size_;
  int channels = this->channels_;
  int height = this->original_size_;
  int width = this->original_size_;
  for (int m = 0; m < 2; ++m) {
    Caffe::set_mode(modes[m]);
    for (int p = 0; p < 2; ++p) {
      Caffe::set_phase(phases[p]);
      for (int t = 0; t < 3; ++t) {
        processor_param.mutable_mirroring_param()->set_type(types[t]);
        for (int r = 0; r < 2; ++r) {
          processor_param.mutable_mirroring_param()->set_random_sampling_ratio(
              random_sampling_ratios[r]);
          MirroringDataProcessor<TypeParam> processor(processor_param);
          shared_ptr<Blob<TypeParam> > blob(new Blob<TypeParam>());
          processor.Process(input_blob, blob);
          EXPECT_EQ(batch_size, blob->num());
          EXPECT_EQ(channels, blob->channels());
          EXPECT_EQ(height, blob->height());
          EXPECT_EQ(width, blob->width());
          is_mirrored = processor.is_mirrored();
          EXPECT_EQ(batch_size, is_mirrored.size());
          num_is_mirrored = std::count(is_mirrored.begin(), is_mirrored.end(),
                                       true);
          const TypeParam* output_data = blob->cpu_data();
          for (int n = 0; n < blob->num(); ++n) {
            if (is_mirrored[n]) {
              switch (types[t]) {
              case MirroringParameter::UP_DOWN: {
                BLOB_DATUM_DIMS_LOOP_BEGIN(channels,  height, width)
                    EXPECT_EQ(
                        data[this->blob_->offset(n, c, h, w)],
                        output_data[blob->offset(n, c, height - 1 - h, w)]) <<
                        "debug: n " << n << " c " << c << " h " << h <<
                        " w " << w << " type " << types[t] <<
                        " random_sampling_ratio " << random_sampling_ratios[r];
                BLOB_DATUM_DIMS_LOOP_END
                break;
              }
              case MirroringParameter::LEFT_RIGHT_AND_UP_DOWN: {
                BLOB_DATUM_DIMS_LOOP_BEGIN(channels,  height, width)
                    EXPECT_EQ(
                        data[this->blob_->offset(n, c, h, w)],
                        output_data[blob->offset(
                            n, c, height - 1 - h, width - 1 - w)]) <<
                            "debug: n " << n << " c " << c << " h " << h <<
                            " w " << w << " type " << types[t] <<
                            " random_sampling_ratio " <<
                            random_sampling_ratios[r];
                BLOB_DATUM_DIMS_LOOP_END
                break;
              }
              case MirroringParameter::LEFT_RIGHT: {
                BLOB_DATUM_DIMS_LOOP_BEGIN(channels,  height, width)
                    EXPECT_EQ(
                        data[this->blob_->offset(n, c, h, w)],
                        output_data[blob->offset(n, c, h, width -1 - w)]) <<
                            "debug: n " << n << " c " << c << " h " << h <<
                            " w " << w << " type " << types[t] <<
                            " random_sampling_ratio " <<
                            random_sampling_ratios[r];
                BLOB_DATUM_DIMS_LOOP_END
                break;
              }
              }
            } else {
              BLOB_DATUM_DIMS_LOOP_BEGIN(channels,  height, width)
                  EXPECT_EQ(
                      data[this->blob_->offset(n, c, h, w)],
                      output_data[blob->offset(n, c, h, w)]) <<
                      "debug: n " << n << " c " << c << " h " << h <<
                      " w " << w << " type " << types[t] <<
                      " random_sampling_ratio " << random_sampling_ratios[r];
              BLOB_DATUM_DIMS_LOOP_END
            }
          }
        }  // for (int r = 0; r < 2; ++r) {
      }  // for (int t = 0; t < 3; ++t) {
    }  // for (int p = 0; p < 2; ++p) {
  }  // for (int m = 0; m < 2; ++m) {
}

TYPED_TEST(DataProcessorTest, TestMeanSubtractionDataProcessor_Process){
  DataProcessorParameter processor_param;
  MeanSubtractionDataProcessor<TypeParam> processor(processor_param);

  shared_ptr<Blob<TypeParam> > input_blob(this->blob_);
  const TypeParam* data = this->blob_->cpu_data();
  Caffe::Brew modes[] = {Caffe::CPU, Caffe::GPU};
  for (int m = 0; m < 2; ++m) {
    Caffe::set_mode(modes[m]);
    shared_ptr<Blob<TypeParam> > mean_blob(
        new Blob<TypeParam>(this->blob_->num(), this->blob_->channels(),
                            this->blob_->height(), this->blob_->width()));
    this->filler_.Fill(mean_blob.get());
    processor.set_mean_blob(mean_blob);
    EXPECT_EQ(mean_blob->cpu_data(), processor.mean_blob_data());
    shared_ptr<Blob<TypeParam> > blob(new Blob<TypeParam>());
    processor.Process(input_blob, blob);
    EXPECT_EQ(this->batch_size_, blob->num());
    EXPECT_EQ(this->channels_, blob->channels());
    EXPECT_EQ(this->original_size_, blob->height());
    EXPECT_EQ(this->original_size_, blob->width());
    BLOB_ALL_DIMS_LOOP_BEGIN(blob->num(), blob->channels(), blob->height(),
                             blob->width())
    EXPECT_EQ(blob->data_at(n, c, h, w),
              this->blob_->data_at(n, c, h, w) -
              mean_blob->data_at(n, c, h, w))
              << "debug: n " << n << " c " << c << " h " << h << " w " << w;
    BLOB_ALL_DIMS_LOOP_END
  }  // for (int m = 0; m < 2; ++m) {
}

TYPED_TEST(DataProcessorTest, TestScalingDataProcessor_Process){
  DataProcessorParameter processor_param;
  ScalingDataProcessor<TypeParam> processor(processor_param);

  shared_ptr<Blob<TypeParam> > input_blob(this->blob_);
  const TypeParam* data = this->blob_->cpu_data();
  Caffe::Brew modes[] = {Caffe::CPU, Caffe::GPU};
  TypeParam scales[] = {-2, -1, -0.5, 0, 0.5, 1, 2};
  for (int m = 0; m < 2; ++m) {
    Caffe::set_mode(modes[m]);
    for (int s = 0; s < 7; ++s) {
      processor.set_scale(scales[s]);
      EXPECT_EQ(scales[s], processor.scale());
      shared_ptr<Blob<TypeParam> > blob(new Blob<TypeParam>());
      processor.Process(input_blob, blob);
      EXPECT_EQ(input_blob->num(), blob->num());
      EXPECT_EQ(input_blob->channels(), blob->channels());
      EXPECT_EQ(input_blob->height(), blob->height());
      EXPECT_EQ(input_blob->width(), blob->width());
      BLOB_ALL_DIMS_LOOP_BEGIN(blob->num(), blob->channels(), blob->height(),
                               blob->width())
      EXPECT_EQ(blob->data_at(n, c, h, w),
                input_blob->data_at(n, c, h, w) * scales[s])
                << "debug: n " << n << " c " << c << " h " << h << " w " << w;
      BLOB_ALL_DIMS_LOOP_END
    }  // for (int s = 0; s < 7; ++s) {
  }  // for (int m = 0; m < 2; ++m) {
}

}
  // namespace caffe
