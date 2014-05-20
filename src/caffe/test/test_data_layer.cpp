// Copyright 2014 BVLC and contributors.

#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "leveldb/db.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/test/test_caffe_main.hpp"

using std::string;
using std::stringstream;

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class DataLayerTest : public ::testing::Test {
 protected:
  DataLayerTest()
      : blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        filename_(new string(tmpnam(NULL))),
        seed_(1701) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
  }

  // Fill the LevelDB with data: if unique_pixels, each pixel is unique but
  // all images are the same; else each image is unique but all pixels within
  // an image are the same.
  void FillLevelDB(const bool unique_pixels) {
    LOG(INFO) << "Using temporary leveldb " << *filename_;
    leveldb::DB* db;
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    leveldb::Status status =
        leveldb::DB::Open(options, filename_->c_str(), &db);
    CHECK(status.ok());
    for (int i = 0; i < 5; ++i) {
      Datum datum;
      datum.set_label(i);
      datum.set_channels(2);
      datum.set_height(3);
      datum.set_width(4);
      std::string* data = datum.mutable_data();
      for (int j = 0; j < 24; ++j) {
        int datum = unique_pixels ? j : i;
        data->push_back(static_cast<uint8_t>(datum));
      }
      stringstream ss;
      ss << i;
      db->Put(leveldb::WriteOptions(), ss.str(), datum.SerializeAsString());
    }
    delete db;
  }

  virtual ~DataLayerTest() { delete blob_top_data_; delete blob_top_label_; }

  shared_ptr<string> filename_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int seed_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(DataLayerTest, Dtypes);

TYPED_TEST(DataLayerTest, TestReadCPU) {
  Caffe::set_mode(Caffe::CPU);
  const bool unique_pixels = false;  // all pixels the same; images different
  this->FillLevelDB(unique_pixels);
  const TypeParam scale = 3;
  LayerParameter param;
  DataParameter* data_param = param.mutable_data_param();
  data_param->set_batch_size(5);
  data_param->set_scale(scale);
  data_param->set_source(this->filename_->c_str());
  DataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 2);
  EXPECT_EQ(this->blob_top_data_->height(), 3);
  EXPECT_EQ(this->blob_top_data_->width(), 4);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  for (int iter = 0; iter < 100; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 24; ++j) {
        EXPECT_EQ(scale * i, this->blob_top_data_->cpu_data()[i * 24 + j])
            << "debug: iter " << iter << " i " << i << " j " << j;
      }
    }
  }
}

TYPED_TEST(DataLayerTest, TestReadGPU) {
  Caffe::set_mode(Caffe::GPU);
  const bool unique_pixels = false;  // all pixels the same; images different
  this->FillLevelDB(unique_pixels);
  const TypeParam scale = 3;
  LayerParameter param;
  DataParameter* data_param = param.mutable_data_param();
  data_param->set_batch_size(5);
  data_param->set_scale(scale);
  data_param->set_source(this->filename_->c_str());
  DataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 2);
  EXPECT_EQ(this->blob_top_data_->height(), 3);
  EXPECT_EQ(this->blob_top_data_->width(), 4);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  for (int iter = 0; iter < 100; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 24; ++j) {
        EXPECT_EQ(scale * i, this->blob_top_data_->cpu_data()[i * 24 + j])
            << "debug: iter " << iter << " i " << i << " j " << j;
      }
    }
  }
}

TYPED_TEST(DataLayerTest, TestReadCropTrainCPU) {
  Caffe::set_phase(Caffe::TRAIN);
  Caffe::set_mode(Caffe::CPU);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLevelDB(unique_pixels);
  const TypeParam scale = 3;
  LayerParameter param;
  DataParameter* data_param = param.mutable_data_param();
  data_param->set_batch_size(5);
  data_param->set_scale(scale);
  data_param->set_crop_size(1);
  data_param->set_source(this->filename_->c_str());
  DataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 2);
  EXPECT_EQ(this->blob_top_data_->height(), 1);
  EXPECT_EQ(this->blob_top_data_->width(), 1);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
    int num_with_center_value = 0;
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 2; ++j) {
        const TypeParam center_value = scale * (j ? 17 : 5);
        num_with_center_value +=
            (center_value == this->blob_top_data_->cpu_data()[i * 2 + j]);
      }
    }
    // Check we did not get the center crop all 10 times (occurs with
    // probability 1-1/12^10 in working implementation).
    EXPECT_LT(num_with_center_value, 10);
  }
}

TYPED_TEST(DataLayerTest, TestReadCropTrainGPU) {
  Caffe::set_phase(Caffe::TRAIN);
  Caffe::set_mode(Caffe::GPU);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLevelDB(unique_pixels);
  const TypeParam scale = 3;
  LayerParameter param;
  DataParameter* data_param = param.mutable_data_param();
  data_param->set_batch_size(5);
  data_param->set_scale(scale);
  data_param->set_crop_size(1);
  data_param->set_source(this->filename_->c_str());
  DataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 2);
  EXPECT_EQ(this->blob_top_data_->height(), 1);
  EXPECT_EQ(this->blob_top_data_->width(), 1);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
    int num_with_center_value = 0;
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 2; ++j) {
        const TypeParam center_value = scale * (j ? 17 : 5);
        num_with_center_value +=
            (center_value == this->blob_top_data_->cpu_data()[i * 2 + j]);
      }
    }
    // Check we did not get the center crop all 10 times (occurs with
    // probability 1-1/12^10 in working implementation).
    EXPECT_LT(num_with_center_value, 10);
  }
}

// Test that the sequence of random crops is consistent when using
// Caffe::set_random_seed.
TYPED_TEST(DataLayerTest, TestReadCropTrainSequenceSeededCPU) {
  Caffe::set_phase(Caffe::TRAIN);
  Caffe::set_mode(Caffe::CPU);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLevelDB(unique_pixels);
  LayerParameter param;
  DataParameter* data_param = param.mutable_data_param();
  data_param->set_batch_size(5);
  data_param->set_crop_size(1);
  data_param->set_mirror(true);
  data_param->set_source(this->filename_->c_str());

  // Get crop sequence with Caffe seed 1701.
  Caffe::set_random_seed(this->seed_);
  vector<vector<TypeParam> > crop_sequence;
  {
    DataLayer<TypeParam> layer1(param);
    layer1.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int iter = 0; iter < 2; ++iter) {
      layer1.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
      for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
      }
      vector<TypeParam> iter_crop_sequence;
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 2; ++j) {
          iter_crop_sequence.push_back(
              this->blob_top_data_->cpu_data()[i * 2 + j]);
        }
      }
      crop_sequence.push_back(iter_crop_sequence);
    }
  }  // destroy 1st data layer and unlock the leveldb

  // Get crop sequence after reseeding Caffe with 1701.
  // Check that the sequence is the same as the original.
  Caffe::set_random_seed(this->seed_);
  DataLayer<TypeParam> layer2(param);
  layer2.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  for (int iter = 0; iter < 2; ++iter) {
    layer2.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 2; ++j) {
        EXPECT_EQ(crop_sequence[iter][i * 2 + j],
                  this->blob_top_data_->cpu_data()[i * 2 + j])
            << "debug: iter " << iter << " i " << i << " j " << j;
      }
    }
  }
}

// Test that the sequence of random crops is consistent when using
// Caffe::set_random_seed.
TYPED_TEST(DataLayerTest, TestReadCropTrainSequenceSeededGPU) {
  Caffe::set_phase(Caffe::TRAIN);
  Caffe::set_mode(Caffe::GPU);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLevelDB(unique_pixels);
  LayerParameter param;
  DataParameter* data_param = param.mutable_data_param();
  data_param->set_batch_size(5);
  data_param->set_crop_size(1);
  data_param->set_mirror(true);
  data_param->set_source(this->filename_->c_str());

  // Get crop sequence with Caffe seed 1701.
  Caffe::set_random_seed(this->seed_);
  vector<vector<TypeParam> > crop_sequence;
  {
    DataLayer<TypeParam> layer1(param);
    layer1.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int iter = 0; iter < 2; ++iter) {
      layer1.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
      for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
      }
      vector<TypeParam> iter_crop_sequence;
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 2; ++j) {
          iter_crop_sequence.push_back(
              this->blob_top_data_->cpu_data()[i * 2 + j]);
        }
      }
      crop_sequence.push_back(iter_crop_sequence);
    }
  }  // destroy 1st data layer and unlock the leveldb

  // Get crop sequence after reseeding Caffe with 1701.
  // Check that the sequence is the same as the original.
  Caffe::set_random_seed(this->seed_);
  DataLayer<TypeParam> layer2(param);
  layer2.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  for (int iter = 0; iter < 2; ++iter) {
    layer2.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 2; ++j) {
        EXPECT_EQ(crop_sequence[iter][i * 2 + j],
                  this->blob_top_data_->cpu_data()[i * 2 + j])
            << "debug: iter " << iter << " i " << i << " j " << j;
      }
    }
  }
}

// Test that the sequence of random crops differs across iterations when
// Caffe::set_random_seed isn't called (and seeds from srand are ignored).
TYPED_TEST(DataLayerTest, TestReadCropTrainSequenceUnseededCPU) {
  Caffe::set_phase(Caffe::TRAIN);
  Caffe::set_mode(Caffe::CPU);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLevelDB(unique_pixels);
  LayerParameter param;
  DataParameter* data_param = param.mutable_data_param();
  data_param->set_batch_size(5);
  data_param->set_crop_size(1);
  data_param->set_mirror(true);
  data_param->set_source(this->filename_->c_str());

  // Get crop sequence with Caffe seed 1701, srand seed 1701.
  Caffe::set_random_seed(this->seed_);
  srand(this->seed_);
  vector<vector<TypeParam> > crop_sequence;
  {
    DataLayer<TypeParam> layer1(param);
    layer1.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int iter = 0; iter < 2; ++iter) {
      layer1.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
      for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
      }
      vector<TypeParam> iter_crop_sequence;
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 2; ++j) {
          iter_crop_sequence.push_back(
              this->blob_top_data_->cpu_data()[i * 2 + j]);
        }
      }
      crop_sequence.push_back(iter_crop_sequence);
    }
  }  // destroy 1st data layer and unlock the leveldb

  // Get crop sequence continuing from previous Caffe RNG state;
  // reseed srand with 1701. Check that the sequence differs from the original.
  srand(this->seed_);
  DataLayer<TypeParam> layer2(param);
  layer2.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  for (int iter = 0; iter < 2; ++iter) {
    layer2.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
    int num_sequence_matches = 0;
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 2; ++j) {
        num_sequence_matches += (crop_sequence[iter][i * 2 + j] ==
                                 this->blob_top_data_->cpu_data()[i * 2 + j]);
      }
    }
    EXPECT_LT(num_sequence_matches, 10);
  }
}

// Test that the sequence of random crops differs across iterations when
// Caffe::set_random_seed isn't called (and seeds from srand are ignored).
TYPED_TEST(DataLayerTest, TestReadCropTrainSequenceUnseededGPU) {
  Caffe::set_phase(Caffe::TRAIN);
  Caffe::set_mode(Caffe::GPU);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLevelDB(unique_pixels);
  LayerParameter param;
  DataParameter* data_param = param.mutable_data_param();
  data_param->set_batch_size(5);
  data_param->set_crop_size(1);
  data_param->set_mirror(true);
  data_param->set_source(this->filename_->c_str());

  // Get crop sequence with Caffe seed 1701, srand seed 1701.
  Caffe::set_random_seed(this->seed_);
  srand(this->seed_);
  vector<vector<TypeParam> > crop_sequence;
  {
    DataLayer<TypeParam> layer1(param);
    layer1.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int iter = 0; iter < 2; ++iter) {
      layer1.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
      for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
      }
      vector<TypeParam> iter_crop_sequence;
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 2; ++j) {
          iter_crop_sequence.push_back(
              this->blob_top_data_->cpu_data()[i * 2 + j]);
        }
      }
      crop_sequence.push_back(iter_crop_sequence);
    }
  }  // destroy 1st data layer and unlock the leveldb

  // Get crop sequence continuing from previous Caffe RNG state;
  // reseed srand with 1701. Check that the sequence differs from the original.
  srand(this->seed_);
  DataLayer<TypeParam> layer2(param);
  layer2.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  for (int iter = 0; iter < 2; ++iter) {
    layer2.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
    int num_sequence_matches = 0;
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 2; ++j) {
        num_sequence_matches += (crop_sequence[iter][i * 2 + j] ==
                                 this->blob_top_data_->cpu_data()[i * 2 + j]);
      }
    }
    EXPECT_LT(num_sequence_matches, 10);
  }
}

TYPED_TEST(DataLayerTest, TestReadCropTestCPU) {
  Caffe::set_phase(Caffe::TEST);
  Caffe::set_mode(Caffe::CPU);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLevelDB(unique_pixels);
  const TypeParam scale = 3;
  LayerParameter param;
  DataParameter* data_param = param.mutable_data_param();
  data_param->set_batch_size(5);
  data_param->set_scale(scale);
  data_param->set_crop_size(1);
  data_param->set_source(this->filename_->c_str());
  DataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 2);
  EXPECT_EQ(this->blob_top_data_->height(), 1);
  EXPECT_EQ(this->blob_top_data_->width(), 1);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 2; ++j) {
        const TypeParam center_value = scale * (j ? 17 : 5);
        EXPECT_EQ(center_value, this->blob_top_data_->cpu_data()[i * 2 + j])
            << "debug: iter " << iter << " i " << i << " j " << j;
      }
    }
  }
}

TYPED_TEST(DataLayerTest, TestReadCropTestGPU) {
  Caffe::set_phase(Caffe::TEST);
  Caffe::set_mode(Caffe::GPU);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLevelDB(unique_pixels);
  const TypeParam scale = 3;
  LayerParameter param;
  DataParameter* data_param = param.mutable_data_param();
  data_param->set_batch_size(5);
  data_param->set_scale(scale);
  data_param->set_crop_size(1);
  data_param->set_source(this->filename_->c_str());
  DataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 2);
  EXPECT_EQ(this->blob_top_data_->height(), 1);
  EXPECT_EQ(this->blob_top_data_->width(), 1);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 2; ++j) {
        const TypeParam center_value = scale * (j ? 17 : 5);
        EXPECT_EQ(center_value, this->blob_top_data_->cpu_data()[i * 2 + j])
            << "debug: iter " << iter << " i " << i << " j " << j;
      }
    }
  }
}

}  // namespace caffe
