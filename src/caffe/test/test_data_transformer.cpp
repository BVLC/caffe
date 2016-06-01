#ifdef USE_OPENCV
#include <algorithm>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class DataTransformTest : public ::testing::Test {
 protected:
  DataTransformTest()
      : seed_(1701),
      num_iter_(10),
      channels_(2),
      height_(10),
      width_(10) {}

  void FillDatum(const int label, const bool unique_pixels, Datum * datum) {
    datum->set_label(label);
    datum->set_channels(channels_);
    datum->set_height(height_);
    datum->set_width(width_);
    int size = channels_ * height_ * width_;
    std::string* data = datum->mutable_data();
    for (int j = 0; j < size; ++j) {
      int datum = unique_pixels ? j : label;
      data->push_back(static_cast<uint8_t>(datum));
    }
  }

  void FillAnnotatedDatum(const int label, const bool unique_pixels,
                          const bool use_rich_annotation,
                          AnnotatedDatum_AnnotationType type,
                          AnnotatedDatum* anno_datum) {
    Datum* datum = anno_datum->mutable_datum();
    // Fill datum.
    datum->set_channels(channels_);
    datum->set_height(height_);
    datum->set_width(width_);
    int size = channels_ * height_ * width_;
    std::string* data = datum->mutable_data();
    for (int j = 0; j < size; ++j) {
      int elem = unique_pixels ? j : label;
      data->push_back(static_cast<uint8_t>(elem));
    }
    // Fill annotation.
    if (use_rich_annotation) {
      anno_datum->set_type(type);
      AnnotationGroup* anno_group = anno_datum->add_annotation_group();
      anno_group->set_group_label(label);
      for (int a = 0; a < 9; ++a) {
        Annotation* anno = anno_group->add_annotation();
        anno->set_instance_id(a);
        if (type == AnnotatedDatum_AnnotationType_BBOX) {
          NormalizedBBox* bbox = anno->mutable_bbox();
          bbox->set_xmin(a*0.1);
          bbox->set_ymin(a*0.1);
          bbox->set_xmax(std::min(a*0.1 + 0.2, 1.0));
          bbox->set_ymax(std::min(a*0.1 + 0.2, 1.0));
        }
      }
    } else {
      datum->set_label(label);
    }
  }

  int NumSequenceMatches(const TransformationParameter transform_param,
      const Datum& datum, Phase phase) {
    // Get crop sequence with Caffe seed 1701.
    DataTransformer<Dtype> transformer(transform_param, phase);
    const int crop_size = transform_param.crop_size();
    Caffe::set_random_seed(seed_);
    transformer.InitRand();
    Blob<Dtype> blob(1, datum.channels(), datum.height(), datum.width());
    if (transform_param.crop_size() > 0) {
      blob.Reshape(1, datum.channels(), crop_size, crop_size);
    }

    vector<vector<Dtype> > crop_sequence;
    for (int iter = 0; iter < this->num_iter_; ++iter) {
      vector<Dtype> iter_crop_sequence;
      transformer.Transform(datum, &blob);
      for (int j = 0; j < blob.count(); ++j) {
        iter_crop_sequence.push_back(blob.cpu_data()[j]);
      }
      crop_sequence.push_back(iter_crop_sequence);
    }
    // Check if the sequence differs from the previous
    int num_sequence_matches = 0;
    for (int iter = 0; iter < this->num_iter_; ++iter) {
      vector<Dtype> iter_crop_sequence = crop_sequence[iter];
      transformer.Transform(datum, &blob);
      for (int j = 0; j < blob.count(); ++j) {
        num_sequence_matches += (crop_sequence[iter][j] == blob.cpu_data()[j]);
      }
    }
    return num_sequence_matches;
  }

  int seed_;
  int num_iter_;
  int channels_;
  int height_;
  int width_;
};

TYPED_TEST_CASE(DataTransformTest, TestDtypes);

TYPED_TEST(DataTransformTest, TestEmptyTransform) {
  TransformationParameter transform_param;
  const bool unique_pixels = false;  // all pixels the same equal to label
  const int label = 0;

  Datum datum;
  this->FillDatum(label, unique_pixels, &datum);
  Blob<TypeParam> blob(1, this->channels_, this->height_, this->width_);
  DataTransformer<TypeParam> transformer(transform_param, TEST);
  transformer.InitRand();
  transformer.Transform(datum, &blob);
  EXPECT_EQ(blob.num(), 1);
  EXPECT_EQ(blob.channels(), datum.channels());
  EXPECT_EQ(blob.height(), datum.height());
  EXPECT_EQ(blob.width(), datum.width());
  for (int j = 0; j < blob.count(); ++j) {
    EXPECT_EQ(blob.cpu_data()[j], label);
  }
}

TYPED_TEST(DataTransformTest, TestEmptyTransformUniquePixels) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;

  Datum datum;
  this->FillDatum(label, unique_pixels, &datum);
  Blob<TypeParam> blob(1, this->channels_, this->height_, this->width_);
  DataTransformer<TypeParam> transformer(transform_param, TEST);
  transformer.InitRand();
  transformer.Transform(datum, &blob);
  EXPECT_EQ(blob.num(), 1);
  EXPECT_EQ(blob.channels(), datum.channels());
  EXPECT_EQ(blob.height(), datum.height());
  EXPECT_EQ(blob.width(), datum.width());
  for (int j = 0; j < blob.count(); ++j) {
    EXPECT_EQ(blob.cpu_data()[j], j);
  }
}

TYPED_TEST(DataTransformTest, TestCropSize) {
  TransformationParameter transform_param;
  const bool unique_pixels = false;  // all pixels the same equal to label
  const int label = 0;
  const int crop_size = 2;

  transform_param.set_crop_size(crop_size);
  Datum datum;
  this->FillDatum(label, unique_pixels, &datum);
  DataTransformer<TypeParam> transformer(transform_param, TEST);
  transformer.InitRand();
  Blob<TypeParam> blob(1, this->channels_, crop_size, crop_size);
  for (int iter = 0; iter < this->num_iter_; ++iter) {
    transformer.Transform(datum, &blob);
    EXPECT_EQ(blob.num(), 1);
    EXPECT_EQ(blob.channels(), datum.channels());
    EXPECT_EQ(blob.height(), crop_size);
    EXPECT_EQ(blob.width(), crop_size);
    for (int j = 0; j < blob.count(); ++j) {
      EXPECT_EQ(blob.cpu_data()[j], label);
    }
  }
}

TYPED_TEST(DataTransformTest, TestCropTrain) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int crop_size = 2;
  const int size = this->channels_ * crop_size * crop_size;

  transform_param.set_crop_size(crop_size);
  Datum datum;
  this->FillDatum(label, unique_pixels, &datum);
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
  EXPECT_LT(num_matches, size * this->num_iter_);
}

TYPED_TEST(DataTransformTest, TestCropTest) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int crop_size = 2;
  const int size = this->channels_ * crop_size * crop_size;

  transform_param.set_crop_size(crop_size);
  Datum datum;
  this->FillDatum(label, unique_pixels, &datum);
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
  EXPECT_EQ(num_matches, size * this->num_iter_);
}

TYPED_TEST(DataTransformTest, TestMirrorTrain) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int size = this->channels_ * this->height_ * this->width_;

  transform_param.set_mirror(true);
  Datum datum;
  this->FillDatum(label, unique_pixels, &datum);
  int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
  EXPECT_LT(num_matches, size * this->num_iter_);
}

TYPED_TEST(DataTransformTest, TestMirrorTest) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int size = this->channels_ * this->height_ * this->width_;

  transform_param.set_mirror(true);
  Datum datum;
  this->FillDatum(label, unique_pixels, &datum);
  int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
  EXPECT_LT(num_matches, size * this->num_iter_);
}

TYPED_TEST(DataTransformTest, TestCropMirrorTrain) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int crop_size = 2;

  Datum datum;
  this->FillDatum(label, unique_pixels, &datum);
  transform_param.set_crop_size(crop_size);
  int num_matches_crop = this->NumSequenceMatches(
      transform_param, datum, TRAIN);

  transform_param.set_mirror(true);
  int num_matches_crop_mirror =
      this->NumSequenceMatches(transform_param, datum, TRAIN);
  // When doing crop and mirror we expect less num_matches than just crop
  EXPECT_LE(num_matches_crop_mirror, num_matches_crop);
}

TYPED_TEST(DataTransformTest, TestCropMirrorTest) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int crop_size = 2;

  Datum datum;
  this->FillDatum(label, unique_pixels, &datum);
  transform_param.set_crop_size(crop_size);
  int num_matches_crop = this->NumSequenceMatches(transform_param, datum, TEST);

  transform_param.set_mirror(true);
  int num_matches_crop_mirror =
      this->NumSequenceMatches(transform_param, datum, TEST);
  // When doing crop and mirror we expect less num_matches than just crop
  EXPECT_LT(num_matches_crop_mirror, num_matches_crop);
}


TYPED_TEST(DataTransformTest, TestMeanValue) {
  TransformationParameter transform_param;
  const bool unique_pixels = false;  // pixels are equal to label
  const int label = 0;
  const int mean_value = 2;

  transform_param.add_mean_value(mean_value);
  Datum datum;
  this->FillDatum(label, unique_pixels, &datum);
  Blob<TypeParam> blob(1, this->channels_, this->height_, this->width_);
  DataTransformer<TypeParam> transformer(transform_param, TEST);
  transformer.InitRand();
  transformer.Transform(datum, &blob);
  for (int j = 0; j < blob.count(); ++j) {
    EXPECT_EQ(blob.cpu_data()[j], label - mean_value);
  }
}

TYPED_TEST(DataTransformTest, TestMeanValues) {
  TransformationParameter transform_param;
  const bool unique_pixels = false;  // pixels are equal to label
  const int label = 0;

  for (int c = 0; c < this->channels_; ++c) {
    transform_param.add_mean_value(c);
  }
  Datum datum;
  this->FillDatum(label, unique_pixels, &datum);
  Blob<TypeParam> blob(1, this->channels_, this->height_, this->width_);
  DataTransformer<TypeParam> transformer(transform_param, TEST);
  transformer.InitRand();
  transformer.Transform(datum, &blob);
  for (int c = 0; c < this->channels_; ++c) {
    for (int j = 0; j < this->height_ * this->width_; ++j) {
      EXPECT_EQ(blob.cpu_data()[blob.offset(0, c) + j], label - c);
    }
  }
}

TYPED_TEST(DataTransformTest, TestMeanFile) {
  TransformationParameter transform_param;
  const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
  const int label = 0;
  const int size = this->channels_ * this->height_ * this->width_;

  // Create a mean file
  string mean_file;
  MakeTempFilename(&mean_file);
  BlobProto blob_mean;
  blob_mean.set_num(1);
  blob_mean.set_channels(this->channels_);
  blob_mean.set_height(this->height_);
  blob_mean.set_width(this->width_);

  for (int j = 0; j < size; ++j) {
      blob_mean.add_data(j);
  }

  LOG(INFO) << "Using temporary mean_file " << mean_file;
  WriteProtoToBinaryFile(blob_mean, mean_file);

  transform_param.set_mean_file(mean_file);
  Datum datum;
  this->FillDatum(label, unique_pixels, &datum);
  Blob<TypeParam> blob(1, this->channels_, this->height_, this->width_);
  DataTransformer<TypeParam> transformer(transform_param, TEST);
  transformer.InitRand();
  transformer.Transform(datum, &blob);
  for (int j = 0; j < blob.count(); ++j) {
      EXPECT_EQ(blob.cpu_data()[j], 0);
  }
}

TYPED_TEST(DataTransformTest, TestRichLabel) {
  TransformationParameter transform_param;
  const bool unique_pixels = false;  // all pixels the same equal to label
  const int label = 0;
  const bool use_rich_annotation = true;
  AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
  const float eps = 1e-6;

  AnnotatedDatum anno_datum;
  this->FillAnnotatedDatum(label, unique_pixels, use_rich_annotation, type,
                           &anno_datum);
  Blob<TypeParam> blob(1, this->channels_, this->height_, this->width_);
  vector<AnnotationGroup> transformed_anno_vec;

  DataTransformer<TypeParam> transformer(transform_param, TEST);
  transformer.InitRand();
  transformer.Transform(anno_datum, &blob, &transformed_anno_vec);

  EXPECT_EQ(transformed_anno_vec.size(), 1);
  AnnotationGroup& anno_group = transformed_anno_vec[0];
  EXPECT_EQ(anno_group.group_label(), label);
  EXPECT_EQ(anno_group.annotation_size(), 9);
  for (int a = 0; a < 9; ++a) {
    const Annotation& anno = anno_group.annotation(a);
    EXPECT_EQ(anno.instance_id(), a);
    EXPECT_NEAR(anno.bbox().xmin(), a*0.1, eps);
    EXPECT_NEAR(anno.bbox().ymin(), a*0.1, eps);
    EXPECT_NEAR(anno.bbox().xmax(), a*0.1 + 0.2, eps);
    EXPECT_NEAR(anno.bbox().ymax(), a*0.1 + 0.2, eps);
  }
}

TYPED_TEST(DataTransformTest, TestRichLabelCrop) {
  TransformationParameter transform_param;
  const bool unique_pixels = false;  // all pixels the same equal to label
  const int label = 0;
  const bool use_rich_annotation = true;
  AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
  const float eps = 1e-6;
  const int crop_size = 1;

  AnnotatedDatum anno_datum;
  this->FillAnnotatedDatum(label, unique_pixels, use_rich_annotation, type,
                           &anno_datum);
  Blob<TypeParam> blob(1, this->channels_, crop_size, crop_size);
  vector<AnnotationGroup> transformed_anno_vec;

  transform_param.set_crop_size(crop_size);
  DataTransformer<TypeParam> transformer(transform_param, TEST);
  transformer.InitRand();
  transformer.Transform(anno_datum, &blob, &transformed_anno_vec);

  EXPECT_EQ(transformed_anno_vec.size(), 1);
  AnnotationGroup& anno_group = transformed_anno_vec[0];
  EXPECT_EQ(anno_group.group_label(), label);
  EXPECT_EQ(anno_group.annotation_size(), 2);
  for (int a = 0; a < anno_group.annotation_size(); ++a) {
    const Annotation& anno = anno_group.annotation(a);
    EXPECT_NEAR(anno.bbox().xmin(), 0., eps);
    EXPECT_NEAR(anno.bbox().ymin(), 0., eps);
    EXPECT_NEAR(anno.bbox().xmax(), 1., eps);
    EXPECT_NEAR(anno.bbox().ymax(), 1., eps);
  }
}

TYPED_TEST(DataTransformTest, TestRichLabelCropMirror) {
  TransformationParameter transform_param;
  const bool unique_pixels = false;  // all pixels the same equal to label
  const int label = 0;
  const bool use_rich_annotation = true;
  AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
  const float eps = 1e-6;
  const int crop_size = 4;

  AnnotatedDatum anno_datum;
  this->FillAnnotatedDatum(label, unique_pixels, use_rich_annotation, type,
                           &anno_datum);
  Blob<TypeParam> blob(1, this->channels_, crop_size, crop_size);

  transform_param.set_crop_size(crop_size);
  transform_param.set_mirror(true);
  DataTransformer<TypeParam> transformer(transform_param, TEST);
  transformer.InitRand();
  bool do_mirror;
  for (int iter = 0; iter < 10; ++iter) {
    vector<AnnotationGroup> transformed_anno_vec;
    transformer.Transform(anno_datum, &blob, &transformed_anno_vec, &do_mirror);

    EXPECT_EQ(transformed_anno_vec.size(), 1);
    AnnotationGroup& anno_group = transformed_anno_vec[0];
    EXPECT_EQ(anno_group.group_label(), label);
    EXPECT_EQ(anno_group.annotation_size(), 5);
    for (int a = 2; a < 7; ++a) {
      const Annotation& anno = anno_group.annotation(a-2);
      if (do_mirror) {
        EXPECT_NEAR(anno.bbox().xmin(), 1. - std::min((a-1), 4)/4., eps);
        EXPECT_NEAR(anno.bbox().ymin(), std::max((a-3), 0)/4., eps);
        EXPECT_NEAR(anno.bbox().xmax(), 1. - std::max((a-3), 0)/4., eps);
        EXPECT_NEAR(anno.bbox().ymax(), std::min((a-1), 4)/4., eps);
      } else {
        EXPECT_NEAR(anno.bbox().xmin(), std::max((a-3), 0)/4., eps);
        EXPECT_NEAR(anno.bbox().ymin(), std::max((a-3), 0)/4., eps);
        EXPECT_NEAR(anno.bbox().xmax(), std::min((a-1), 4)/4., eps);
        EXPECT_NEAR(anno.bbox().ymax(), std::min((a-1), 4)/4., eps);
      }
    }
  }
}

}  // namespace caffe
#endif  // USE_OPENCV
