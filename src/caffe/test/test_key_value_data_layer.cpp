#ifdef USE_OPENCV
#include <string>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/key_value_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

using boost::scoped_ptr;

template <typename TypeParam>
class KeyValueDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  KeyValueDataLayerTest()
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

  // Fill the DB with data: if unique_pixels, each pixel is unique but
  // all images are the same; else each image is unique but all pixels within
  // an image are the same.
  void Fill(DataParameter_DB backend) {
    backend_ = backend;
    LOG(INFO) << "Using temporary dataset " << *filename_;
    scoped_ptr<db::DB> db(db::GetDB(backend));
    db->Open(*filename_, db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());
    for (int i = 0; i < 5; ++i) {
      Datum datum;
      datum.set_label(i);
      datum.set_channels(2);
      datum.set_height(3);
      datum.set_width(4);
      std::string* data = datum.mutable_data();
      for (int j = 0; j < 24; ++j) {
        data->push_back(static_cast<uint8_t>(i));
      }
      stringstream ss;
      ss << i;
      string out;
      CHECK(datum.SerializeToString(&out));
      txn->Put(ss.str(), out);
    }
    txn->Commit();
    db->Close();
  }

  void FillFile() {
    // Create test input file.
    MakeTempFilename(&keyfilename_);
    std::ofstream outfile(keyfilename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << keyfilename_;
    for (int i = 0; i < 5; ++i) {
      outfile << i << ";" << 4-i << ";" << 1 << '\n';
    }
    outfile.close();
  }

  void TestRead(const vector<int>& keys, int column) {
    const Dtype scale = 3;
    LayerParameter param;
    param.set_phase(TRAIN);
    DataParameter* data_param = param.mutable_data_param();
    data_param->set_batch_size(5);
    data_param->set_source(filename_->c_str());
    data_param->set_backend(backend_);

    TransformationParameter* transform_param =
        param.mutable_transform_param();
    transform_param->set_scale(scale);

    KeyValueDataParameter* kvdata_param = param.mutable_key_value_data_param();
    kvdata_param->set_key_file(keyfilename_.c_str());
    kvdata_param->set_column(column);

    KeyValueDataLayer<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_data_->num(), 5);
    EXPECT_EQ(blob_top_data_->channels(), 2);
    EXPECT_EQ(blob_top_data_->height(), 3);
    EXPECT_EQ(blob_top_data_->width(), 4);
    EXPECT_EQ(blob_top_label_->num(), 5);
    EXPECT_EQ(blob_top_label_->channels(), 1);
    EXPECT_EQ(blob_top_label_->height(), 1);
    EXPECT_EQ(blob_top_label_->width(), 1);

    for (int iter = 0; iter < 100; ++iter) {
      layer.Forward(blob_bottom_vec_, blob_top_vec_);
      for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(keys[i], blob_top_label_->cpu_data()[i]);
      }
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 24; ++j) {
          EXPECT_EQ(scale * keys[i], blob_top_data_->cpu_data()[i * 24 + j])
              << "debug: iter " << iter << " i " << i << " j " << j;
        }
      }
    }
  }
  virtual ~KeyValueDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  DataParameter_DB backend_;
  shared_ptr<string> filename_;
  string keyfilename_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int seed_;
};

TYPED_TEST_CASE(KeyValueDataLayerTest, TestDtypesAndDevices);
#ifdef USE_LEVELDB
TYPED_TEST(KeyValueDataLayerTest, TestReadLevelDB) {
  this->Fill(DataParameter_DB_LEVELDB);
  this->FillFile();
  vector<int> keys(5);

  for (int i = 0; i < 5; ++i)
    keys[i] = i;
  this->TestRead(keys, 1);

  for (int i = 0; i < 5; ++i)
    keys[i] = 4-i;
  this->TestRead(keys, 2);

  for (int i = 0; i < 5; ++i)
    keys[i] = 1;
  this->TestRead(keys, 3);
}
#endif  // USE_LEVELDB

#ifdef USE_LMDB
TYPED_TEST(KeyValueDataLayerTest, TestReadLMDB) {
  this->Fill(DataParameter_DB_LMDB);
  this->FillFile();
  vector<int> keys(5);

  for (int i = 0; i < 5; ++i)
    keys[i] = i;
  this->TestRead(keys, 1);

  for (int i = 0; i < 5; ++i)
    keys[i] = 4-i;
  this->TestRead(keys, 2);

  for (int i = 0; i < 5; ++i)
    keys[i] = 1;
  this->TestRead(keys, 3);
}
#endif  // USE_LMDB
}  // namespace caffe
#endif  // USE_OPENCV
