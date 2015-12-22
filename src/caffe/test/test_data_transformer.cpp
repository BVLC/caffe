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
#ifdef USE_OPENCV
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
#ifdef USE_OPENCV
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
#ifdef USE_OPENCV
=======
>>>>>>> pod/device/blob.hpp
=======
#ifdef USE_OPENCV
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifdef USE_OPENCV
=======
>>>>>>> pod/caffe-merge
=======
#ifdef USE_OPENCV
=======
>>>>>>> pod/device/blob.hpp
=======
#ifdef USE_OPENCV
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifdef USE_OPENCV
=======
>>>>>>> pod/device/blob.hpp
=======
#ifdef USE_OPENCV
=======
>>>>>>> pod/device/blob.hpp
=======
#ifdef USE_OPENCV
=======
>>>>>>> pod/device/blob.hpp
=======
#ifdef USE_OPENCV
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> origin/BVLC/parallel
=======
#ifdef USE_OPENCV
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
#ifdef USE_OPENCV
>>>>>>> device-abstraction
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
#ifdef USE_OPENCV
>>>>>>> device-abstraction
=======
#ifdef USE_OPENCV
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "leveldb/db.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

void FillDatum(const int label, const int channels, const int height,
  const int width, const bool unique_pixels, Datum * datum) {
  datum->set_label(label);
  datum->set_channels(channels);
  datum->set_height(height);
  datum->set_width(width);
  int size = channels * height * width;
  std::string* data = datum->mutable_data();
  for (int j = 0; j < size; ++j) {
    int datum = unique_pixels ? j : label;
    data->push_back(static_cast<uint8_t>(datum));
  }
}

template <typename Dtype>
class DataTransformTest : public ::testing::Test {
 protected:
  DataTransformTest()
      : seed_(1701),
      num_iter_(10) {}

  int NumSequenceMatches(const TransformationParameter transform_param,
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
      const Datum& datum, Phase phase) {
    // Get crop sequence with Caffe seed 1701.
    DataTransformer<Dtype>* transformer =
        new DataTransformer<Dtype>(transform_param, phase);
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
      const Datum& datum) {
    // Get crop sequence with Caffe seed 1701.
    DataTransformer<Dtype>* transformer =
        new DataTransformer<Dtype>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
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
>>>>>>> device-abstraction
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
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
    const int crop_size = transform_param.crop_size();
    Caffe::set_random_seed(seed_);
    transformer->InitRand();
    Blob<Dtype>* blob =
        new Blob<Dtype>(1, datum.channels(), datum.height(), datum.width());
    if (transform_param.crop_size() > 0) {
      blob->Reshape(1, datum.channels(), crop_size, crop_size);
    }

    vector<vector<Dtype> > crop_sequence;
    for (int iter = 0; iter < this->num_iter_; ++iter) {
      vector<Dtype> iter_crop_sequence;
      transformer->Transform(datum, blob);
      for (int j = 0; j < blob->count(); ++j) {
        iter_crop_sequence.push_back(blob->cpu_data()[j]);
      }
      crop_sequence.push_back(iter_crop_sequence);
    }
    // Check if the sequence differs from the previous
    int num_sequence_matches = 0;
    for (int iter = 0; iter < this->num_iter_; ++iter) {
      vector<Dtype> iter_crop_sequence = crop_sequence[iter];
      transformer->Transform(datum, blob);
      for (int j = 0; j < blob->count(); ++j) {
        num_sequence_matches +=
            (crop_sequence[iter][j] == blob->cpu_data()[j]);
      }
    }
    return num_sequence_matches;
  }

  virtual ~DataTransformTest() { }

  int seed_;
  int num_iter_;
};

TYPED_TEST_CASE(DataTransformTest, TestDtypes);

TYPED_TEST(DataTransformTest, TestEmptyTransform) {
  TransformationParameter transform_param;
  const bool unique_pixels = false;  // all pixels the same equal to label
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;

  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
  Blob<TypeParam>* blob = new Blob<TypeParam>(1, channels, height, width);
  DataTransformer<TypeParam>* transformer =
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
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/caffe-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/device/blob.hpp
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
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
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> device-abstraction
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
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> device-abstraction
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> pod/post-rebase-error-fix
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> caffe
  transformer->InitRand();
  transformer->Transform(datum, blob);
  EXPECT_EQ(blob->num(), 1);
  EXPECT_EQ(blob->channels(), datum.channels());
  EXPECT_EQ(blob->height(), datum.height());
  EXPECT_EQ(blob->width(), datum.width());
  for (int j = 0; j < blob->count(); ++j) {
    EXPECT_EQ(blob->cpu_data()[j], label);
  }
}

TYPED_TEST(DataTransformTest, TestEmptyTransformUniquePixels) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;

  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
  Blob<TypeParam>* blob = new Blob<TypeParam>(1, 3, 4, 5);
  DataTransformer<TypeParam>* transformer =
<<<<<<< HEAD
<<<<<<< HEAD
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  transformer->InitRand();
  transformer->Transform(datum, blob);
  EXPECT_EQ(blob->num(), 1);
  EXPECT_EQ(blob->channels(), datum.channels());
  EXPECT_EQ(blob->height(), datum.height());
  EXPECT_EQ(blob->width(), datum.width());
  for (int j = 0; j < blob->count(); ++j) {
<<<<<<< HEAD
    EXPECT_EQ(blob->cpu_data()[j], label);
  }
}

TYPED_TEST(DataTransformTest, TestEmptyTransformUniquePixels) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
=======
    EXPECT_EQ(blob->cpu_data()[j], j);
  }
}

TYPED_TEST(DataTransformTest, TestCropSize) {
  TransformationParameter transform_param;
  const bool unique_pixels = false;  // all pixels the same equal to label
>>>>>>> pod-caffe-pod.hpp-merge
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
<<<<<<< HEAD

  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
  Blob<TypeParam>* blob = new Blob<TypeParam>(1, 3, 4, 5);
  DataTransformer<TypeParam>* transformer =
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
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/caffe-merge
=======
  const int crop_size = 2;

  transform_param.set_crop_size(crop_size);
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
  DataTransformer<TypeParam>* transformer =
<<<<<<< HEAD
<<<<<<< HEAD
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod-caffe-pod.hpp-merge
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> caffe
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
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> device-abstraction
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> pod/post-rebase-error-fix
  transformer->InitRand();
  transformer->Transform(datum, blob);
  EXPECT_EQ(blob->num(), 1);
  EXPECT_EQ(blob->channels(), datum.channels());
  EXPECT_EQ(blob->height(), datum.height());
  EXPECT_EQ(blob->width(), datum.width());
  for (int j = 0; j < blob->count(); ++j) {
    EXPECT_EQ(blob->cpu_data()[j], j);
  }
}

TYPED_TEST(DataTransformTest, TestCropSize) {
  TransformationParameter transform_param;
  const bool unique_pixels = false;  // all pixels the same equal to label
=======
  transformer->InitRand();
  Blob<TypeParam>* blob =
      new Blob<TypeParam>(1, channels, crop_size, crop_size);
  for (int iter = 0; iter < this->num_iter_; ++iter) {
    transformer->Transform(datum, blob);
    EXPECT_EQ(blob->num(), 1);
    EXPECT_EQ(blob->channels(), datum.channels());
    EXPECT_EQ(blob->height(), crop_size);
    EXPECT_EQ(blob->width(), crop_size);
    for (int j = 0; j < blob->count(); ++j) {
      EXPECT_EQ(blob->cpu_data()[j], label);
    }
  }
}

TYPED_TEST(DataTransformTest, TestCropTrain) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  const int crop_size = 2;
  const int size = channels * crop_size * crop_size;

  transform_param.set_crop_size(crop_size);
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
<<<<<<< HEAD
<<<<<<< HEAD
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
  Caffe::set_phase(Caffe::TRAIN);
  int num_matches = this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
>>>>>>> caffe
  EXPECT_LT(num_matches, size * this->num_iter_);
}

TYPED_TEST(DataTransformTest, TestCropTest) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  const int crop_size = 2;
  const int size = channels * crop_size * crop_size;

  transform_param.set_crop_size(crop_size);
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
<<<<<<< HEAD
<<<<<<< HEAD
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
  Caffe::set_phase(Caffe::TEST);
  int num_matches = this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
>>>>>>> caffe
  EXPECT_EQ(num_matches, size * this->num_iter_);
}

TYPED_TEST(DataTransformTest, TestMirrorTrain) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  const int size = channels * height * width;

  transform_param.set_mirror(true);
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
<<<<<<< HEAD
<<<<<<< HEAD
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
  Caffe::set_phase(Caffe::TRAIN);
  int num_matches = this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
>>>>>>> caffe
  EXPECT_LT(num_matches, size * this->num_iter_);
}

TYPED_TEST(DataTransformTest, TestMirrorTest) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  const int size = channels * height * width;

  transform_param.set_mirror(true);
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
<<<<<<< HEAD
<<<<<<< HEAD
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
  Caffe::set_phase(Caffe::TEST);
  int num_matches = this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
>>>>>>> caffe
  EXPECT_LT(num_matches, size * this->num_iter_);
}

TYPED_TEST(DataTransformTest, TestCropMirrorTrain) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
>>>>>>> pod-caffe-pod.hpp-merge
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  const int crop_size = 2;

<<<<<<< HEAD
  transform_param.set_crop_size(crop_size);
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
  DataTransformer<TypeParam>* transformer =
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
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/caffe-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/device/blob.hpp
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
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
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> device-abstraction
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
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> device-abstraction
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> pod/post-rebase-error-fix
  transformer->InitRand();
  Blob<TypeParam>* blob =
      new Blob<TypeParam>(1, channels, crop_size, crop_size);
  for (int iter = 0; iter < this->num_iter_; ++iter) {
    transformer->Transform(datum, blob);
    EXPECT_EQ(blob->num(), 1);
    EXPECT_EQ(blob->channels(), datum.channels());
    EXPECT_EQ(blob->height(), crop_size);
    EXPECT_EQ(blob->width(), crop_size);
    for (int j = 0; j < blob->count(); ++j) {
      EXPECT_EQ(blob->cpu_data()[j], label);
    }
  }
}

TYPED_TEST(DataTransformTest, TestCropTrain) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  const int crop_size = 2;
  const int size = channels * crop_size * crop_size;

  transform_param.set_crop_size(crop_size);
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
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
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
>>>>>>> pod/device/blob.hpp
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
>>>>>>> pod/caffe-merge
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
>>>>>>> pod/device/blob.hpp
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
>>>>>>> pod/device/blob.hpp
  Caffe::set_phase(Caffe::TRAIN);
  int num_matches = this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
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
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
  Caffe::set_phase(Caffe::TRAIN);
  int num_matches = this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
  Caffe::set_phase(Caffe::TRAIN);
  int num_matches = this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
>>>>>>> device-abstraction
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
>>>>>>> pod/post-rebase-error-fix
  EXPECT_LT(num_matches, size * this->num_iter_);
}

TYPED_TEST(DataTransformTest, TestCropTest) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  const int crop_size = 2;
  const int size = channels * crop_size * crop_size;

  transform_param.set_crop_size(crop_size);
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
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
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
>>>>>>> pod/device/blob.hpp
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  Caffe::set_phase(Caffe::TEST);
  int num_matches = this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
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
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
>>>>>>> device-abstraction
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
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
>>>>>>> device-abstraction
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
>>>>>>> pod/post-rebase-error-fix
  EXPECT_EQ(num_matches, size * this->num_iter_);
}

TYPED_TEST(DataTransformTest, TestMirrorTrain) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  const int size = channels * height * width;

  transform_param.set_mirror(true);
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
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
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
>>>>>>> pod/device/blob.hpp
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
>>>>>>> pod/device/blob.hpp
  Caffe::set_phase(Caffe::TRAIN);
  int num_matches = this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
>>>>>>> caffe
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
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
  Caffe::set_phase(Caffe::TRAIN);
  int num_matches = this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
  Caffe::set_phase(Caffe::TRAIN);
  int num_matches = this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
  Caffe::set_phase(Caffe::TRAIN);
  int num_matches = this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
=======
  Caffe::set_phase(Caffe::TRAIN);
  int num_matches = this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
>>>>>>> device-abstraction
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
>>>>>>> pod/post-rebase-error-fix
  EXPECT_LT(num_matches, size * this->num_iter_);
}

TYPED_TEST(DataTransformTest, TestMirrorTest) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  const int size = channels * height * width;

  transform_param.set_mirror(true);
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
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
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
>>>>>>> pod/device/blob.hpp
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
>>>>>>> pod/caffe-merge
  Caffe::set_phase(Caffe::TEST);
  int num_matches = this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
>>>>>>> caffe
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
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  Caffe::set_phase(Caffe::TEST);
  int num_matches = this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
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
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
>>>>>>> device-abstraction
=======
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
>>>>>>> pod/post-rebase-error-fix
  EXPECT_LT(num_matches, size * this->num_iter_);
}

TYPED_TEST(DataTransformTest, TestCropMirrorTrain) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  const int crop_size = 2;

  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
  transform_param.set_crop_size(crop_size);
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
  int num_matches_crop = this->NumSequenceMatches(
      transform_param, datum, TRAIN);

  transform_param.set_mirror(true);
  int num_matches_crop_mirror =
      this->NumSequenceMatches(transform_param, datum, TRAIN);
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
  Caffe::set_phase(Caffe::TRAIN);
  int num_matches_crop = this->NumSequenceMatches(transform_param, datum);

  transform_param.set_mirror(true);
  int num_matches_crop_mirror =
      this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
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
>>>>>>> device-abstraction
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
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
  // When doing crop and mirror we expect less num_matches than just crop
  EXPECT_LE(num_matches_crop_mirror, num_matches_crop);
}

TYPED_TEST(DataTransformTest, TestCropMirrorTest) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  const int crop_size = 2;

  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
  transform_param.set_crop_size(crop_size);
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
  int num_matches_crop = this->NumSequenceMatches(transform_param, datum, TEST);

  transform_param.set_mirror(true);
  int num_matches_crop_mirror =
      this->NumSequenceMatches(transform_param, datum, TEST);
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
  Caffe::set_phase(Caffe::TEST);
  int num_matches_crop = this->NumSequenceMatches(transform_param, datum);

  transform_param.set_mirror(true);
  int num_matches_crop_mirror =
      this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
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
>>>>>>> device-abstraction
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
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
  // When doing crop and mirror we expect less num_matches than just crop
  EXPECT_LT(num_matches_crop_mirror, num_matches_crop);
}


TYPED_TEST(DataTransformTest, TestMeanValue) {
  TransformationParameter transform_param;
  const bool unique_pixels = false;  // pixels are equal to label
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  const int mean_value = 2;

  transform_param.add_mean_value(mean_value);
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
  Blob<TypeParam>* blob = new Blob<TypeParam>(1, channels, height, width);
  DataTransformer<TypeParam>* transformer =
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
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/caffe-merge
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> caffe
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
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
=======
=======
>>>>>>> pod/device/blob.hpp
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
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
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> device-abstraction
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> pod/post-rebase-error-fix
  transformer->InitRand();
  transformer->Transform(datum, blob);
  for (int j = 0; j < blob->count(); ++j) {
    EXPECT_EQ(blob->cpu_data()[j], label - mean_value);
  }
}

TYPED_TEST(DataTransformTest, TestMeanValues) {
  TransformationParameter transform_param;
  const bool unique_pixels = false;  // pixels are equal to label
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;

  transform_param.add_mean_value(0);
  transform_param.add_mean_value(1);
  transform_param.add_mean_value(2);
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
  Blob<TypeParam>* blob = new Blob<TypeParam>(1, channels, height, width);
  DataTransformer<TypeParam>* transformer =
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
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/caffe-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod-caffe-pod.hpp-merge
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> caffe
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
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
=======
=======
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
  transform_param.set_crop_size(crop_size);
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
  int num_matches_crop = this->NumSequenceMatches(
      transform_param, datum, TRAIN);

  transform_param.set_mirror(true);
  int num_matches_crop_mirror =
      this->NumSequenceMatches(transform_param, datum, TRAIN);
<<<<<<< HEAD
=======
  Caffe::set_phase(Caffe::TRAIN);
  int num_matches_crop = this->NumSequenceMatches(transform_param, datum);

  transform_param.set_mirror(true);
  int num_matches_crop_mirror =
      this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
  // When doing crop and mirror we expect less num_matches than just crop
  EXPECT_LE(num_matches_crop_mirror, num_matches_crop);
}

TYPED_TEST(DataTransformTest, TestCropMirrorTest) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  const int crop_size = 2;

  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
  transform_param.set_crop_size(crop_size);
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
  int num_matches_crop = this->NumSequenceMatches(transform_param, datum, TEST);

  transform_param.set_mirror(true);
  int num_matches_crop_mirror =
      this->NumSequenceMatches(transform_param, datum, TEST);
<<<<<<< HEAD
=======
  Caffe::set_phase(Caffe::TEST);
  int num_matches_crop = this->NumSequenceMatches(transform_param, datum);

  transform_param.set_mirror(true);
  int num_matches_crop_mirror =
      this->NumSequenceMatches(transform_param, datum);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
  // When doing crop and mirror we expect less num_matches than just crop
  EXPECT_LT(num_matches_crop_mirror, num_matches_crop);
}


TYPED_TEST(DataTransformTest, TestMeanValue) {
  TransformationParameter transform_param;
  const bool unique_pixels = false;  // pixels are equal to label
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  const int mean_value = 2;

  transform_param.add_mean_value(mean_value);
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
  Blob<TypeParam>* blob = new Blob<TypeParam>(1, channels, height, width);
  DataTransformer<TypeParam>* transformer =
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
  transformer->InitRand();
  transformer->Transform(datum, blob);
  for (int j = 0; j < blob->count(); ++j) {
    EXPECT_EQ(blob->cpu_data()[j], label - mean_value);
  }
}

TYPED_TEST(DataTransformTest, TestMeanValues) {
  TransformationParameter transform_param;
  const bool unique_pixels = false;  // pixels are equal to label
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;

  transform_param.add_mean_value(0);
  transform_param.add_mean_value(1);
  transform_param.add_mean_value(2);
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
  Blob<TypeParam>* blob = new Blob<TypeParam>(1, channels, height, width);
  DataTransformer<TypeParam>* transformer =
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> device-abstraction
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
  transformer->InitRand();
  transformer->Transform(datum, blob);
  for (int c = 0; c < channels; ++c) {
    for (int j = 0; j < height * width; ++j) {
      EXPECT_EQ(blob->cpu_data()[blob->offset(0, c) + j], label - c);
    }
  }
}

TYPED_TEST(DataTransformTest, TestMeanFile) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  const int size = channels * height * width;

  // Create a mean file
  string* mean_file = new string();
  MakeTempFilename(mean_file);
  BlobProto blob_mean;
  blob_mean.set_num(1);
  blob_mean.set_channels(channels);
  blob_mean.set_height(height);
  blob_mean.set_width(width);

  for (int j = 0; j < size; ++j) {
      blob_mean.add_data(j);
  }

  LOG(INFO) << "Using temporary mean_file " << *mean_file;
  WriteProtoToBinaryFile(blob_mean, *mean_file);

  transform_param.set_mean_file(*mean_file);
  Datum datum;
  FillDatum(label, channels, height, width, unique_pixels, &datum);
  Blob<TypeParam>* blob = new Blob<TypeParam>(1, channels, height, width);
  DataTransformer<TypeParam>* transformer =
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
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/caffe-merge
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
<<<<<<< HEAD
<<<<<<< HEAD
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod/device/blob.hpp
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
=======
>>>>>>> pod-caffe-pod.hpp-merge
      new DataTransformer<TypeParam>(transform_param);
>>>>>>> origin/BVLC/parallel
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
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
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> device-abstraction
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
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> device-abstraction
=======
      new DataTransformer<TypeParam>(transform_param, TEST);
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
  transformer->InitRand();
  transformer->Transform(datum, blob);
  for (int j = 0; j < blob->count(); ++j) {
      EXPECT_EQ(blob->cpu_data()[j], 0);
  }
}

}  // namespace caffe
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
#endif  // USE_OPENCV
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
#endif  // USE_OPENCV
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
#endif  // USE_OPENCV
=======
>>>>>>> pod/device/blob.hpp
=======
#endif  // USE_OPENCV
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
#endif  // USE_OPENCV
=======
>>>>>>> pod/caffe-merge
>>>>>>> origin/BVLC/parallel
=======
#endif  // USE_OPENCV
>>>>>>> caffe
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
#endif  // USE_OPENCV
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/device/blob.hpp
#endif  // USE_OPENCV
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
#endif  // USE_OPENCV
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
#endif  // USE_OPENCV
=======
>>>>>>> origin/BVLC/parallel
=======
#endif  // USE_OPENCV
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
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
#endif  // USE_OPENCV
>>>>>>> device-abstraction
=======
#endif  // USE_OPENCV
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
