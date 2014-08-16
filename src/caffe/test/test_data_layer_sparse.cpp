#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/dataset_factory.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/sparse_blob.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template<typename TypeParam>
class DataLayerSparseTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DataLayerSparseTest()
      : backend_(DataParameter_DB_LEVELDB),
        blob_top_data_(new SparseBlob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        seed_(1701) {
  }
  virtual void SetUp() {
    filename_.reset(new string());
    MakeTempDir(filename_.get());
    *filename_ += "/db";
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
  }

  void Fill(DataParameter_DB backend) {
    backend_ = backend;
    LOG(INFO) << "Using temporary dataset " << *filename_
        << " with backend: " << backend;
    shared_ptr<Dataset<string, SparseDatum> > dataset = DatasetFactory<string,
        SparseDatum>(backend_);
    CHECK(dataset->open(*filename_, Dataset<string, SparseDatum>::New));
    for (int i = 0; i < 6; ++i) {
      SparseDatum datum;
      datum.set_label(i);
      datum.set_size(6);
      datum.set_nn(i + 1);
      for (int j = 0; j < i + 1; ++j) {
        datum.mutable_data()->Add(j + 1);
        datum.mutable_indices()->Add(j);
      }
      stringstream ss;
      ss << i;
      CHECK(dataset->put(ss.str(), datum));
    }
    CHECK(dataset->commit());
    dataset->close();
  }

  void TestRead() {
    LayerParameter param;
    DataSparseInputParameter* data_param =
        param.mutable_data_sparse_input_param();
    data_param->set_batch_size(6);
    data_param->set_backend(backend_);
    data_param->set_source(filename_->c_str());
    DataLayerSparseInput<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_data_->num(), 6);
    EXPECT_EQ(blob_top_data_->channels(), 6);
    EXPECT_EQ(blob_top_data_->height(), 1);
    EXPECT_EQ(blob_top_data_->width(), 1);
    EXPECT_EQ(blob_top_label_->num(), 6);
    EXPECT_EQ(blob_top_label_->channels(), 1);
    EXPECT_EQ(blob_top_label_->height(), 1);
    EXPECT_EQ(blob_top_label_->width(), 1);

    for (int iter = 0; iter < 100; ++iter) {
      layer.Forward(blob_bottom_vec_, blob_top_vec_);
      for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(i, blob_top_label_->cpu_data()[i]);
      }
      EXPECT_EQ(0, blob_top_data_->cpu_ptr()[0]);
      for (int i = 0; i < 6; ++i) {
        EXPECT_EQ((i+1) * (i+2)/2,
            blob_top_data_->cpu_ptr()[i+1]) << "debug ptr: iter " << iter
                                            << " i " << i;
        for (int j = 0; j < i; ++j) {
          EXPECT_EQ(j+1, blob_top_data_->
              cpu_data()[blob_top_data_->cpu_ptr()[i]+j]) << "debug data: iter "
                                                          << iter << " i " << i
                                                          << " j " << j;
          EXPECT_EQ(j, blob_top_data_->
              cpu_indices()[blob_top_data_->cpu_ptr()[i]+j])
              << "debug indices: iter " << iter << " i " << i << " j " << j;
        }
      }
    }
  }
  void TestRead2() {
    LayerParameter param;
    DataSparseInputParameter* data_param =
        param.mutable_data_sparse_input_param();
    // half the previous batch size to alternate between 2 different dataset
    data_param->set_batch_size(3);
    data_param->set_backend(backend_);
    data_param->set_source(filename_->c_str());
    DataLayerSparseInput<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_data_->num(), 3);
    EXPECT_EQ(blob_top_data_->channels(), 6);
    EXPECT_EQ(blob_top_data_->height(), 1);
    EXPECT_EQ(blob_top_data_->width(), 1);
    EXPECT_EQ(blob_top_label_->num(), 3);
    EXPECT_EQ(blob_top_label_->channels(), 1);
    EXPECT_EQ(blob_top_label_->height(), 1);
    EXPECT_EQ(blob_top_label_->width(), 1);

    int delta = 0;
    for (int iter = 0; iter < 100; ++iter) {
      layer.Forward(blob_bottom_vec_, blob_top_vec_);
      if (iter % 2) {
        delta = 3;
      } else {
        delta = 0;
      }
      for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(i + delta, blob_top_label_->cpu_data()[i]);
      }

      EXPECT_EQ(0, blob_top_data_->cpu_ptr()[0]);
      if (delta == 0) {
        EXPECT_EQ(1, blob_top_data_->cpu_ptr()[1]);
        EXPECT_EQ(3, blob_top_data_->cpu_ptr()[2]);
        EXPECT_EQ(6, blob_top_data_->cpu_ptr()[3]);
      } else {
        EXPECT_EQ(4, blob_top_data_->cpu_ptr()[1]);
        EXPECT_EQ(9, blob_top_data_->cpu_ptr()[2]);
        EXPECT_EQ(15, blob_top_data_->cpu_ptr()[3]);
      }
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i + delta; ++j) {
          EXPECT_EQ(j+1,
              blob_top_data_->cpu_data()[blob_top_data_->cpu_ptr()[i]+j])
              << "debug data: iter " << iter << " i " << i << " j " << j;
          EXPECT_EQ(j,
              blob_top_data_->cpu_indices()[blob_top_data_->cpu_ptr()[i]+j])
              << "debug indices: iter " << iter << " i " << i << " j " << j;
        }
      }
    }
  }

  virtual ~DataLayerSparseTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  DataParameter_DB backend_;
  shared_ptr<string> filename_;
  SparseBlob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int seed_;
};

TYPED_TEST_CASE(DataLayerSparseTest, TestDtypesAndDevices);

TYPED_TEST(DataLayerSparseTest, TestReadLevelDB) {
this->Fill(DataParameter_DB_LEVELDB);
this->TestRead();
}

TYPED_TEST(DataLayerSparseTest, TestReadLevelDB2) {
this->Fill(DataParameter_DB_LEVELDB);
this->TestRead2();
}

TYPED_TEST(DataLayerSparseTest, TestReadLMDB) {
this->Fill(DataParameter_DB_LMDB);
this->TestRead();
}

TYPED_TEST(DataLayerSparseTest, TestReadLMDB2) {
this->Fill(DataParameter_DB_LMDB);
this->TestRead2();
}

}  // namespace caffe
