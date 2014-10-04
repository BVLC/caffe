#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "leveldb/db.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class MapDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MapDataLayerTest()
      : backend_(DataParameter_DB_LEVELDB),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        seed_(1701) {}
  virtual void SetUp() {
    filename_.reset(new string());
    MakeTempDir(filename_.get());
    *filename_ += "/db";
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
  }

  // Fill the LevelDB with data: if unique_pixels, each pixel is unique but
  // all images are the same; else each image is unique but all pixels within
  // an image are the same.
  void FillLevelDB(const bool unique_pixels) {
    backend_ = DataParameter_DB_LEVELDB;
    LOG(INFO) << "Using temporary leveldb " << *filename_;
    leveldb::DB* db;
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    leveldb::Status status =
        leveldb::DB::Open(options, filename_->c_str(), &db);
    CHECK(status.ok());
    for (int i = 0; i < 5; ++i) {
      BlobProtoVector sample;
      BlobProto* dataMap = sample.add_blobs();
      dataMap->set_channels(2);
      dataMap->set_height(3);
      dataMap->set_width(4);
      for (int j = 0; j < 24; ++j) {
        int datum = unique_pixels ? j : i;
        dataMap->add_data(datum);
      }

      BlobProto* labelMap = sample.add_blobs();
      labelMap->set_channels(1);
      labelMap->set_height(3);
      labelMap->set_width(4);
      for (int label_idx = 0; label_idx < 12; ++label_idx){
        labelMap->add_data(i * 12 + label_idx);
      }

      stringstream ss;
      ss << i;
      db->Put(leveldb::WriteOptions(), ss.str(), sample.SerializeAsString());
    }
    delete db;
  }

  void TestRead() {
    const Dtype scale = 3;
    LayerParameter param;
    DataParameter* data_param = param.mutable_data_param();
    data_param->set_batch_size(5);
    data_param->set_source(filename_->c_str());
    data_param->set_backend(backend_);

    TransformationParameter* transform_param =
        param.mutable_transform_param();
    transform_param->set_scale(scale);

    MapDataLayer<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_data_->num(), 5);
    EXPECT_EQ(blob_top_data_->channels(), 2);
    EXPECT_EQ(blob_top_data_->height(), 3);
    EXPECT_EQ(blob_top_data_->width(), 4);
    EXPECT_EQ(blob_top_label_->num(), 5);
    EXPECT_EQ(blob_top_label_->channels(), 1);
    EXPECT_EQ(blob_top_label_->height(), 3);
    EXPECT_EQ(blob_top_label_->width(), 4);

    for (int iter = 0; iter < 100; ++iter) {
      layer.Forward(blob_bottom_vec_, &blob_top_vec_);
      for (int i = 0; i < 5; ++i) {
        for (int label_idx = 0; label_idx < 12; ++label_idx) {
          int top_idx = i * 12 + label_idx;
          EXPECT_EQ(top_idx, blob_top_label_->cpu_data()[top_idx])
              << "debug: iter" << iter << " i " << i << " j " << label_idx;
        }
      }
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 24; ++j) {
          EXPECT_EQ(scale * i, blob_top_data_->cpu_data()[i * 24 + j])
              << "debug: iter " << iter << " i " << i << " j " << j;
        }
      }
    }
  }

  virtual ~MapDataLayerTest() { delete blob_top_data_; delete blob_top_label_; }

  DataParameter_DB backend_;
  shared_ptr<string> filename_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int seed_;
};

TYPED_TEST_CASE(MapDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(MapDataLayerTest, TestReadLevelDB) {
  const bool unique_pixels = false;  // all pixels the same; images different
  this->FillLevelDB(unique_pixels);
  this->TestRead();
}


}  // namespace caffe
