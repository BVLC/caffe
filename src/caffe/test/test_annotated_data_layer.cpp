#ifdef USE_OPENCV
#include <algorithm>
#include <string>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

using boost::scoped_ptr;

static bool kBoolChoices[] = {false, true};
static int kNumChoices = 2;

// Compute bounding box number.
int OneBBoxNum(int n) {
  int sum = 0;
  for (int g = 0; g < n; ++g) {
    sum += g;
  }
  return sum;
}

int BBoxNum(int n) {
  int sum = 0;
  for (int i = 0; i < n; ++i) {
    for (int g = 0; g < i; ++g) {
      sum += g;
    }
  }
  return sum;
}

template <typename TypeParam>
class AnnotatedDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  AnnotatedDataLayerTest()
      : backend_(DataParameter_DB_LEVELDB),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        seed_(1701),
        num_(6),
        channels_(2),
        height_(10),
        width_(10),
        eps_(1e-6) {}

  virtual void SetUp() {
    spatial_dim_ = height_ * width_;
    size_ = channels_ * spatial_dim_;
    filename_.reset(new string());
    GetTempDirname(filename_.get());
    *filename_ += "/db";
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
  }

  // Fill the DB with data.
  //  - backend: can be either LevelDB or LMDB
  //  - unique_pixel: if true, each pixel is unique but all images are the same;
  //  else each image is unique but all pixels within an image are the same.
  //  - unique_annotation: if true, each annotation in a group is unique but all
  //  groups are the same at the same positions; else each group is unique but
  //  all annotations within a group are the same.
  //  - use_rich_annotation: if false, use datum.label() instead.
  //  - type: type of rich annotation.
  void Fill(DataParameter_DB backend, bool unique_pixel, bool unique_annotation,
            bool use_rich_annotation, AnnotatedDatum_AnnotationType type) {
    backend_ = backend;
    unique_pixel_ = unique_pixel;
    unique_annotation_ = unique_annotation;
    use_rich_annotation_ = use_rich_annotation;
    type_ = type;
    GetTempDirname(filename_.get());
    LOG(INFO) << "Using temporary dataset " << *filename_;
    scoped_ptr<db::DB> db(db::GetDB(backend));
    db->Open(*filename_, db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());
    for (int i = 0; i < num_; ++i) {
      AnnotatedDatum anno_datum;
      // Fill data.
      Datum* datum = anno_datum.mutable_datum();
      datum->set_channels(channels_);
      datum->set_height(height_);
      datum->set_width(width_);
      std::string* data = datum->mutable_data();
      for (int j = 0; j < size_; ++j) {
        int elem = unique_pixel ? j : i;
        data->push_back(static_cast<uint8_t>(elem));
      }
      // Fill annotation.
      if (use_rich_annotation) {
        anno_datum.set_type(type);
        for (int g = 0; g < i; ++g) {
          AnnotationGroup* anno_group = anno_datum.add_annotation_group();
          anno_group->set_group_label(g);
          for (int a = 0; a < g; ++a) {
            Annotation* anno = anno_group->add_annotation();
            anno->set_instance_id(a);
            if (type == AnnotatedDatum_AnnotationType_BBOX) {
              NormalizedBBox* bbox = anno->mutable_bbox();
              int b = unique_annotation ? a : g;
              bbox->set_xmin(b*0.1);
              bbox->set_ymin(b*0.1);
              bbox->set_xmax(std::min(b*0.1 + 0.2, 1.0));
              bbox->set_ymax(std::min(b*0.1 + 0.2, 1.0));
              bbox->set_difficult(a % 2);
            }
          }
        }
      } else {
        datum->set_label(i);
      }
      stringstream ss;
      ss << i;
      string out;
      CHECK(anno_datum.SerializeToString(&out));
      txn->Put(ss.str(), out);
    }
    txn->Commit();
    db->Close();
  }

  void TestRead() {
    LayerParameter param;
    param.set_phase(TRAIN);
    DataParameter* data_param = param.mutable_data_param();
    data_param->set_batch_size(num_);
    data_param->set_source(filename_->c_str());
    data_param->set_backend(backend_);

    const Dtype scale = 3;
    TransformationParameter* transform_param =
        param.mutable_transform_param();
    transform_param->set_scale(scale);

    AnnotatedDataLayer<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_data_->num(), num_);
    EXPECT_EQ(blob_top_data_->channels(), channels_);
    EXPECT_EQ(blob_top_data_->height(), height_);
    EXPECT_EQ(blob_top_data_->width(), width_);
    if (use_rich_annotation_) {
      switch (type_) {
        case AnnotatedDatum_AnnotationType_BBOX:
          EXPECT_EQ(blob_top_label_->num(), 1);
          EXPECT_EQ(blob_top_label_->channels(), 1);
          EXPECT_EQ(blob_top_label_->height(), 1);
          EXPECT_EQ(blob_top_label_->width(), 8);
          break;
        default:
          LOG(FATAL) << "Unknown annotation type.";
          break;
      }
    } else {
      EXPECT_EQ(blob_top_label_->num(), num_);
      EXPECT_EQ(blob_top_label_->channels(), 1);
      EXPECT_EQ(blob_top_label_->height(), 1);
      EXPECT_EQ(blob_top_label_->width(), 1);
    }

    for (int iter = 0; iter < 5; ++iter) {
      layer.Forward(blob_bottom_vec_, blob_top_vec_);
      // Check label.
      const Dtype* label_data = blob_top_label_->cpu_data();
      int cur_bbox = 0;
      for (int i = 0; i < num_; ++i) {
        if (use_rich_annotation_) {
          if (type_ == AnnotatedDatum_AnnotationType_BBOX) {
            EXPECT_EQ(blob_top_label_->num(), 1);
            EXPECT_EQ(blob_top_label_->channels(), 1);
            EXPECT_EQ(blob_top_label_->height(), BBoxNum(num_));
            EXPECT_EQ(blob_top_label_->width(), 8);
            for (int g = 0; g < i; ++g) {
              for (int a = 0; a < g; ++a) {
                EXPECT_EQ(i, label_data[cur_bbox*8]);
                EXPECT_EQ(g, label_data[cur_bbox*8+1]);
                EXPECT_EQ(a, label_data[cur_bbox*8+2]);
                int b = unique_annotation_ ? a : g;
                for (int p = 3; p < 5; ++p) {
                  EXPECT_NEAR(b*0.1, label_data[cur_bbox*8+p], this->eps_);
                }
                for (int p = 5; p < 7; ++p) {
                  EXPECT_NEAR(std::min(b*0.1 + 0.2, 1.0),
                            label_data[cur_bbox*8+p], this->eps_);
                }
                EXPECT_EQ(a % 2, label_data[cur_bbox*8+7]);
                cur_bbox++;
              }
            }
          } else {
            LOG(FATAL) << "Unknown annotation type.";
          }
        } else {
          EXPECT_EQ(i, label_data[i]);
        }
      }
      // Check data.
      for (int i = 1; i < num_; ++i) {
        for (int j = 0; j < size_; ++j) {
          EXPECT_EQ(scale * (unique_pixel_ ? j : i),
                    blob_top_data_->cpu_data()[i * size_ + j])
              << "debug: iter " << iter << " i " << i << " j " << j;
        }
      }
    }
  }

  void TestReshape(DataParameter_DB backend, bool unique_pixel,
                   bool unique_annotation, bool use_rich_annotation,
                   AnnotatedDatum_AnnotationType type) {
    // Save data of varying shapes.
    GetTempDirname(filename_.get());
    LOG(INFO) << "Using temporary dataset " << *filename_;
    scoped_ptr<db::DB> db(db::GetDB(backend));
    db->Open(*filename_, db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());
    for (int i = 0; i < num_; ++i) {
      AnnotatedDatum anno_datum;
      // Fill data.
      Datum* datum = anno_datum.mutable_datum();
      datum->set_channels(channels_);
      datum->set_height(i % 2 + 1);
      datum->set_width(i % 4 + 1);
      std::string* data = datum->mutable_data();
      const int data_size =
          datum->channels() * datum->height() * datum->width();
      for (int j = 0; j < data_size; ++j) {
        data->push_back(static_cast<uint8_t>(j));
      }
      // Fill annotation.
      if (use_rich_annotation) {
        anno_datum.set_type(type);
        for (int g = 0; g < i; ++g) {
          AnnotationGroup* anno_group = anno_datum.add_annotation_group();
          anno_group->set_group_label(g);
          for (int a = 0; a < g; ++a) {
            Annotation* anno = anno_group->add_annotation();
            anno->set_instance_id(a);
            if (type == AnnotatedDatum_AnnotationType_BBOX) {
              NormalizedBBox* bbox = anno->mutable_bbox();
              int b = unique_annotation ? a : g;
              bbox->set_xmin(b*0.1);
              bbox->set_ymin(b*0.1);
              bbox->set_xmax(std::min(b*0.1 + 0.2, 1.0));
              bbox->set_ymax(std::min(b*0.1 + 0.2, 1.0));
              bbox->set_difficult(a % 2);
            }
          }
        }
      } else {
        datum->set_label(i);
      }
      stringstream ss;
      ss << i;
      string out;
      CHECK(anno_datum.SerializeToString(&out));
      txn->Put(ss.str(), out);
    }
    txn->Commit();
    db->Close();

    // Load and check data of various shapes.
    LayerParameter param;
    param.set_phase(TEST);
    DataParameter* data_param = param.mutable_data_param();
    data_param->set_batch_size(1);
    data_param->set_source(filename_->c_str());
    data_param->set_backend(backend);

    AnnotatedDataLayer<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_data_->num(), 1);
    EXPECT_EQ(blob_top_data_->channels(), channels_);
    if (use_rich_annotation) {
      switch (type) {
        case AnnotatedDatum_AnnotationType_BBOX:
          EXPECT_EQ(blob_top_label_->num(), 1);
          EXPECT_EQ(blob_top_label_->channels(), 1);
          EXPECT_EQ(blob_top_label_->height(), 1);
          EXPECT_EQ(blob_top_label_->width(), 8);
          break;
        default:
          LOG(FATAL) << "Unknown annotation type.";
          break;
      }
    } else {
      EXPECT_EQ(blob_top_label_->num(), 1);
      EXPECT_EQ(blob_top_label_->channels(), 1);
      EXPECT_EQ(blob_top_label_->height(), 1);
      EXPECT_EQ(blob_top_label_->width(), 1);
    }

    for (int iter = 0; iter < 3; ++iter) {
      layer.Forward(blob_bottom_vec_, blob_top_vec_);
      EXPECT_EQ(blob_top_data_->height(), iter % 2 + 1);
      EXPECT_EQ(blob_top_data_->width(), iter % 4 + 1);
      // Check label.
      const Dtype* label_data = blob_top_label_->cpu_data();
      if (use_rich_annotation) {
        if (type == AnnotatedDatum_AnnotationType_BBOX) {
          if (iter <= 1) {
            EXPECT_EQ(blob_top_label_->num(), 1);
            EXPECT_EQ(blob_top_label_->channels(), 1);
            EXPECT_EQ(blob_top_label_->height(), 1);
            EXPECT_EQ(blob_top_label_->width(), 8);
            for (int i = 0; i < 8; ++i) {
              EXPECT_NEAR(label_data[i], -1, this->eps_);
            }
          } else {
            int cur_bbox = 0;
            EXPECT_EQ(blob_top_label_->num(), 1);
            EXPECT_EQ(blob_top_label_->channels(), 1);
            EXPECT_EQ(blob_top_label_->height(), OneBBoxNum(iter));
            EXPECT_EQ(blob_top_label_->width(), 8);
            for (int g = 0; g < iter; ++g) {
              for (int a = 0; a < g; ++a) {
                EXPECT_EQ(0, label_data[cur_bbox*8]);
                EXPECT_EQ(g, label_data[cur_bbox*8+1]);
                EXPECT_EQ(a, label_data[cur_bbox*8+2]);
                int b = unique_annotation ? a : g;
                for (int p = 3; p < 5; ++p) {
                  EXPECT_NEAR(b*0.1, label_data[cur_bbox*8+p], this->eps_);
                }
                for (int p = 5; p < 7; ++p) {
                  EXPECT_NEAR(std::min(b*0.1 + 0.2, 1.0),
                            label_data[cur_bbox*8+p], this->eps_);
                }
                EXPECT_EQ(a % 2, label_data[cur_bbox*8+7]);
                cur_bbox++;
              }
            }
          }
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
      } else {
        EXPECT_EQ(iter, label_data[0]);
      }
      // Check data.
      const int channels = blob_top_data_->channels();
      const int height = blob_top_data_->height();
      const int width = blob_top_data_->width();
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            const int idx = (c * height + h) * width + w;
            EXPECT_EQ(idx, static_cast<int>(blob_top_data_->cpu_data()[idx]))
                << "debug: iter " << iter << " c " << c
                << " h " << h << " w " << w;
          }
        }
      }
    }
  }

  void TestReadCrop(Phase phase) {
    const Dtype scale = 3;
    LayerParameter param;
    param.set_phase(phase);
    Caffe::set_random_seed(1701);

    DataParameter* data_param = param.mutable_data_param();
    data_param->set_batch_size(num_);
    data_param->set_source(filename_->c_str());
    data_param->set_backend(backend_);

    TransformationParameter* transform_param =
        param.mutable_transform_param();
    transform_param->set_scale(scale);
    transform_param->set_crop_size(1);

    AnnotatedDataLayer<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_data_->num(), num_);
    EXPECT_EQ(blob_top_data_->channels(), channels_);
    EXPECT_EQ(blob_top_data_->height(), 1);
    EXPECT_EQ(blob_top_data_->width(), 1);
    EXPECT_EQ(blob_top_label_->num(), num_);
    EXPECT_EQ(blob_top_label_->channels(), 1);
    EXPECT_EQ(blob_top_label_->height(), 1);
    EXPECT_EQ(blob_top_label_->width(), 1);

    for (int iter = 0; iter < 5; ++iter) {
      layer.Forward(blob_bottom_vec_, blob_top_vec_);
      for (int i = 0; i < num_; ++i) {
        EXPECT_EQ(i, blob_top_label_->cpu_data()[i]);
      }
      int num_with_center_value = 0;
      for (int i = 0; i < num_; ++i) {
        for (int j = 0; j < channels_; ++j) {
          const Dtype center_value =
              scale * ((ceil(height_ / 2.0) - 1) * width_ +
                       ceil(width_ / 2.0) - 1 + j * spatial_dim_);
          num_with_center_value +=
              (center_value == blob_top_data_->cpu_data()[i * 2 + j]);
          // At TEST time, check that we always get center value.
          if (phase == caffe::TEST) {
            EXPECT_EQ(center_value,
                      this->blob_top_data_->cpu_data()[i * channels_ + j])
                << "debug: iter " << iter << " i " << i << " j " << j;
          }
        }
      }
      // At TRAIN time, check that we did not get the center crop all 10 times.
      // (This check fails with probability 1-1/12^10 in a correct
      // implementation, so we call set_random_seed.)
      if (phase == caffe::TRAIN) {
        EXPECT_LT(num_with_center_value, 10);
      }
    }
  }

  void TestReadCropTrainSequenceSeeded() {
    LayerParameter param;
    param.set_phase(TRAIN);
    DataParameter* data_param = param.mutable_data_param();
    data_param->set_batch_size(num_);
    data_param->set_source(filename_->c_str());
    data_param->set_backend(backend_);

    TransformationParameter* transform_param =
        param.mutable_transform_param();
    transform_param->set_crop_size(1);
    transform_param->set_mirror(true);

    // Get crop sequence with Caffe seed 1701.
    Caffe::set_random_seed(seed_);
    vector<vector<Dtype> > crop_sequence;
    {
      AnnotatedDataLayer<Dtype> layer1(param);
      layer1.SetUp(blob_bottom_vec_, blob_top_vec_);
      for (int iter = 0; iter < 2; ++iter) {
        layer1.Forward(blob_bottom_vec_, blob_top_vec_);
        for (int i = 0; i < num_; ++i) {
          EXPECT_EQ(i, blob_top_label_->cpu_data()[i]);
        }
        vector<Dtype> iter_crop_sequence;
        for (int i = 0; i < num_; ++i) {
          for (int j = 0; j < channels_; ++j) {
            iter_crop_sequence.push_back(
                blob_top_data_->cpu_data()[i * channels_ + j]);
          }
        }
        crop_sequence.push_back(iter_crop_sequence);
      }
    }  // destroy 1st data layer and unlock the db

    // Get crop sequence after reseeding Caffe with 1701.
    // Check that the sequence is the same as the original.
    Caffe::set_random_seed(seed_);
    AnnotatedDataLayer<Dtype> layer2(param);
    layer2.SetUp(blob_bottom_vec_, blob_top_vec_);
    for (int iter = 0; iter < 2; ++iter) {
      layer2.Forward(blob_bottom_vec_, blob_top_vec_);
      for (int i = 0; i < num_; ++i) {
        EXPECT_EQ(i, blob_top_label_->cpu_data()[i]);
      }
      for (int i = 0; i < num_; ++i) {
        for (int j = 0; j < channels_; ++j) {
          EXPECT_EQ(crop_sequence[iter][i * channels_ + j],
                    blob_top_data_->cpu_data()[i * channels_ + j])
              << "debug: iter " << iter << " i " << i << " j " << j;
        }
      }
    }
  }

  void TestReadCropTrainSequenceUnseeded() {
    LayerParameter param;
    param.set_phase(TRAIN);
    DataParameter* data_param = param.mutable_data_param();
    data_param->set_batch_size(num_);
    data_param->set_source(filename_->c_str());
    data_param->set_backend(backend_);

    TransformationParameter* transform_param =
        param.mutable_transform_param();
    transform_param->set_crop_size(1);
    transform_param->set_mirror(true);

    // Get crop sequence with Caffe seed 1701, srand seed 1701.
    Caffe::set_random_seed(seed_);
    srand(seed_);
    vector<vector<Dtype> > crop_sequence;
    {
      AnnotatedDataLayer<Dtype> layer1(param);
      layer1.SetUp(blob_bottom_vec_, blob_top_vec_);
      for (int iter = 0; iter < 2; ++iter) {
        layer1.Forward(blob_bottom_vec_, blob_top_vec_);
        for (int i = 0; i < num_; ++i) {
          EXPECT_EQ(i, blob_top_label_->cpu_data()[i]);
        }
        vector<Dtype> iter_crop_sequence;
        for (int i = 0; i < num_; ++i) {
          for (int j = 0; j < channels_; ++j) {
            iter_crop_sequence.push_back(
                blob_top_data_->cpu_data()[i * channels_ + j]);
          }
        }
        crop_sequence.push_back(iter_crop_sequence);
      }
    }  // destroy 1st data layer and unlock the db

    // Get crop sequence continuing from previous Caffe RNG state; reseed
    // srand with 1701. Check that the sequence differs from the original.
    srand(seed_);
    AnnotatedDataLayer<Dtype> layer2(param);
    layer2.SetUp(blob_bottom_vec_, blob_top_vec_);
    for (int iter = 0; iter < 2; ++iter) {
      layer2.Forward(blob_bottom_vec_, blob_top_vec_);
      for (int i = 0; i < num_; ++i) {
        EXPECT_EQ(i, blob_top_label_->cpu_data()[i]);
      }
      int num_sequence_matches = 0;
      for (int i = 0; i < num_; ++i) {
        for (int j = 0; j < channels_; ++j) {
          num_sequence_matches +=
              (crop_sequence[iter][i * channels_ + j] ==
               blob_top_data_->cpu_data()[i * channels_ + j]);
        }
      }
      EXPECT_LT(num_sequence_matches, num_ * channels_);
    }
  }

  virtual ~AnnotatedDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  DataParameter_DB backend_;
  shared_ptr<string> filename_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int seed_;
  int num_;
  int channels_;
  int height_;
  int width_;
  Dtype eps_;
  int spatial_dim_;
  int size_;
  bool unique_pixel_;
  bool unique_annotation_;
  bool use_rich_annotation_;
  AnnotatedDatum_AnnotationType type_;
};

TYPED_TEST_CASE(AnnotatedDataLayerTest, TestDtypesAndDevices);

#ifdef USE_LEVELDB
TYPED_TEST(AnnotatedDataLayerTest, TestReadLevelDB) {
  const AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
  for (int p = 0; p < kNumChoices; ++p) {
    bool unique_pixel = kBoolChoices[p];
    for (int r = 0; r < kNumChoices; ++r) {
      bool use_rich_annotation = kBoolChoices[r];
      for (int a = 0; a < kNumChoices; ++a) {
        if (!use_rich_annotation) {
          continue;
        }
        bool unique_annotation = kBoolChoices[a];
        this->Fill(DataParameter_DB_LEVELDB, unique_pixel, unique_annotation,
                   use_rich_annotation, type);
        this->TestRead();
      }
    }
  }
}

TYPED_TEST(AnnotatedDataLayerTest, TestReshapeLevelDB) {
  const AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
  for (int p = 0; p < kNumChoices; ++p) {
    bool unique_pixel = kBoolChoices[p];
    for (int r = 0; r < kNumChoices; ++r) {
      bool use_rich_annotation = kBoolChoices[r];
      for (int a = 0; a < kNumChoices; ++a) {
        if (!use_rich_annotation) {
          continue;
        }
        bool unique_annotation = kBoolChoices[a];
        this->TestReshape(DataParameter_DB_LEVELDB, unique_pixel,
                          unique_annotation, use_rich_annotation, type);
      }
    }
  }
}

TYPED_TEST(AnnotatedDataLayerTest, TestReadCropTrainLevelDB) {
  const bool unique_pixel = true;  // all pixels the same; images different
  const bool unique_annotation = false;  // all anno the same; groups different
  const bool use_rich_annotation = false;
  AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
  this->Fill(DataParameter_DB_LEVELDB, unique_pixel, unique_annotation,
             use_rich_annotation, type);
  this->TestReadCrop(TRAIN);
}

// Test that the sequence of random crops is consistent when using
// Caffe::set_random_seed.
TYPED_TEST(AnnotatedDataLayerTest, TestReadCropTrainSequenceSeededLevelDB) {
  const bool unique_pixel = true;  // all pixels the same; images different
  const bool unique_annotation = false;  // all anno the same; groups different
  const bool use_rich_annotation = false;
  AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
  this->Fill(DataParameter_DB_LEVELDB, unique_pixel, unique_annotation,
             use_rich_annotation, type);
  this->TestReadCropTrainSequenceSeeded();
}

// Test that the sequence of random crops differs across iterations when
// Caffe::set_random_seed isn't called (and seeds from srand are ignored).
TYPED_TEST(AnnotatedDataLayerTest, TestReadCropTrainSequenceUnseededLevelDB) {
  const bool unique_pixel = true;  // all pixels the same; images different
  const bool unique_annotation = false;  // all anno the same; groups different
  const bool use_rich_annotation = false;
  AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
  this->Fill(DataParameter_DB_LEVELDB, unique_pixel, unique_annotation,
             use_rich_annotation, type);
  this->TestReadCropTrainSequenceUnseeded();
}

TYPED_TEST(AnnotatedDataLayerTest, TestReadCropTestLevelDB) {
  const bool unique_pixel = true;  // all pixels the same; images different
  const bool unique_annotation = false;  // all anno the same; groups different
  const bool use_rich_annotation = false;
  AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
  this->Fill(DataParameter_DB_LEVELDB, unique_pixel, unique_annotation,
             use_rich_annotation, type);
  this->TestReadCrop(TEST);
}
#endif  // USE_LEVELDB

#ifdef USE_LMDB
TYPED_TEST(AnnotatedDataLayerTest, TestReadLMDB) {
  const AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
  for (int p = 0; p < kNumChoices; ++p) {
    bool unique_pixel = kBoolChoices[p];
    for (int r = 0; r < kNumChoices; ++r) {
      bool use_rich_annotation = kBoolChoices[r];
      for (int a = 0; a < kNumChoices; ++a) {
        if (!use_rich_annotation) {
          continue;
        }
        bool unique_annotation = kBoolChoices[a];
        this->Fill(DataParameter_DB_LMDB, unique_pixel, unique_annotation,
                   use_rich_annotation, type);
        this->TestRead();
      }
    }
  }
}

TYPED_TEST(AnnotatedDataLayerTest, TestReshapeLMDB) {
  const AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
  for (int p = 0; p < kNumChoices; ++p) {
    bool unique_pixel = kBoolChoices[p];
    for (int r = 0; r < kNumChoices; ++r) {
      bool use_rich_annotation = kBoolChoices[r];
      for (int a = 0; a < kNumChoices; ++a) {
        if (!use_rich_annotation) {
          continue;
        }
        bool unique_annotation = kBoolChoices[a];
        this->TestReshape(DataParameter_DB_LMDB, unique_pixel,
                          unique_annotation, use_rich_annotation, type);
      }
    }
  }
}

TYPED_TEST(AnnotatedDataLayerTest, TestReadCropTrainLMDB) {
  const bool unique_pixel = true;  // all pixels the same; images different
  const bool unique_annotation = false;  // all anno the same; groups different
  const bool use_rich_annotation = false;
  AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
  this->Fill(DataParameter_DB_LMDB, unique_pixel, unique_annotation,
             use_rich_annotation, type);
  this->TestReadCrop(TRAIN);
}

// Test that the sequence of random crops is consistent when using
// Caffe::set_random_seed.
TYPED_TEST(AnnotatedDataLayerTest, TestReadCropTrainSequenceSeededLMDB) {
  const bool unique_pixel = true;  // all pixels the same; images different
  const bool unique_annotation = false;  // all anno the same; groups different
  const bool use_rich_annotation = false;
  AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
  this->Fill(DataParameter_DB_LMDB, unique_pixel, unique_annotation,
             use_rich_annotation, type);
  this->TestReadCropTrainSequenceSeeded();
}

// Test that the sequence of random crops differs across iterations when
// Caffe::set_random_seed isn't called (and seeds from srand are ignored).
TYPED_TEST(AnnotatedDataLayerTest, TestReadCropTrainSequenceUnseededLMDB) {
  const bool unique_pixel = true;  // all pixels the same; images different
  const bool unique_annotation = false;  // all anno the same; groups different
  const bool use_rich_annotation = false;
  AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
  this->Fill(DataParameter_DB_LMDB, unique_pixel, unique_annotation,
             use_rich_annotation, type);
  this->TestReadCropTrainSequenceUnseeded();
}

TYPED_TEST(AnnotatedDataLayerTest, TestReadCropTestLMDB) {
  const bool unique_pixel = true;  // all pixels the same; images different
  const bool unique_annotation = false;  // all anno the same; groups different
  const bool use_rich_annotation = false;
  AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
  this->Fill(DataParameter_DB_LMDB, unique_pixel, unique_annotation,
             use_rich_annotation, type);
  this->TestReadCrop(TEST);
}

#endif  // USE_LMDB
}  // namespace caffe
#endif  // USE_OPENCV
