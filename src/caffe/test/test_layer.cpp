// Copyright 2014 kloudkl@github

#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
using std::vector;
using std::string;

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template<typename Dtype>
class LayerTest : public ::testing::Test {
 protected:
  LayerTest()
      : max_batch_size_(256),
        channels_(3),
        height_(5),
        width_(7),
        blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);

    layer_types_.push_back("accuracy");
    layer_types_.push_back("bnll");
//    layer_types_.push_back("conv");
    layer_types_.push_back("concat");
//    layer_types_.push_back("data");
//    layer_types_.push_back("hdf5_data");
    layer_types_.push_back("dropout");
    layer_types_.push_back("euclidean_loss");
//    layer_types_.push_back("flatten");
//    layer_types_.push_back("im2col");
//    layer_types_.push_back("infogain_loss");
    layer_types_.push_back("innerproduct");
//    layer_types_.push_back("lrn");
//    layer_types_.push_back("pool");
    layer_types_.push_back("relu");
    layer_types_.push_back("tanh");
    layer_types_.push_back("sigmoid");
    layer_types_.push_back("softmax");
//    layer_types_.push_back("softmax_loss");
    layer_types_.push_back("split");
    layer_types_.push_back("multinomial_logistic_loss");

    // Create the leveldb
    filename_ = tmpnam(NULL);  // get temp name
    LOG(INFO) << "Using temporary leveldb " << filename_;
    leveldb::DB* db;
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    leveldb::Status status = leveldb::DB::Open(options, filename_, &db);
    CHECK(status.ok());
    for (int i = 0; i < 5; ++i) {
      Datum datum;
      datum.set_label(i);
      datum.set_channels(2);
      datum.set_height(3);
      datum.set_width(4);
      std::string* data = datum.mutable_data();
      for (int j = 0; j < 24; ++j) {
        data->push_back((uint8_t)i);
      }
      db->Put(leveldb::WriteOptions(), boost::lexical_cast<string>(i),
              datum.SerializeAsString());
    }
    delete db;
  }
  virtual ~LayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_data_;
    delete blob_top_label_;
  }

  void resize_batch_size(const LayerParameter& layer_param, const int batch_size,
              Filler<Dtype>& filler, vector<Blob<Dtype>*>& blob_vec);

 protected:
  int max_batch_size_;
  int channels_;
  int height_;
  int width_;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<string> layer_types_;
  char* filename_;
};

template<typename Dtype>
void LayerTest<Dtype>::resize_batch_size(const LayerParameter& layer_param,
                                         const int batch_size,
                                         Filler<Dtype>& filler,
                                         vector<Blob<Dtype>*>& blob_vec) {
  for (int j = 0; j < blob_vec.size(); ++j) {
    int channels = channels_;
    int height = height_;
    int width = width_;
    if (layer_param.type() == "accuracy" ||
        layer_param.type().find("loss") != string::npos) {
      channels = 1;
      height = 1;
      width = 1;
    }
    blob_vec[j]->Reshape(batch_size, channels, height, width);
//    LOG(ERROR) << "blob_vec " << j << " batch size " << batch_size <<
//        " channels " << channels << " height " << height << " width " << width;
    filler.Fill(blob_vec[j]);
  }
}

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(LayerTest, Dtypes);

TYPED_TEST(LayerTest, TestDynamicBatchSize){
  Caffe::Brew modes[] = {Caffe::CPU, Caffe::GPU};
  for (int n_mode = 0; n_mode < 2; ++n_mode) {
    Caffe::set_mode(modes[n_mode]);
    LayerParameter layer_param;
    Layer<TypeParam>* layer;
    FillerParameter filler_param;
    filler_param.set_std(10);
    UniformFiller<TypeParam> filler(filler_param);
    int bottom_batch_size;
    int top_batch_size;
    for (size_t i = 0; i < this->layer_types_.size(); ++i) {
      LOG(ERROR) << this->layer_types_[i];
      layer_param.set_type(this->layer_types_[i]);
      layer_param.set_num_output(10);
      layer_param.set_kernelsize(3);
      layer_param.set_concat_dim(0);
      if (layer_param.type() == "data") {
        layer_param.set_batchsize(5);
        layer_param.set_source(this->filename_);
      }

      layer = GetLayer<TypeParam>(layer_param);
      vector<Blob<TypeParam>*> blob_bottom_vec = this->blob_bottom_vec_;
      vector<Blob<TypeParam>*> blob_top_vec = this->blob_top_vec_;
      for (int j = blob_bottom_vec.size(); j > layer->expected_bottom_size();
          --j) {
        blob_bottom_vec.pop_back();
      }
      for (int j = blob_top_vec.size(); j > layer->expected_top_size(); --j) {
        blob_top_vec.pop_back();
      }
      this->resize_batch_size(layer_param, this->max_batch_size_, filler, blob_bottom_vec);
      this->resize_batch_size(layer_param, this->max_batch_size_, filler, blob_top_vec);
      layer->SetUp(blob_bottom_vec, &blob_top_vec);
      for (int n = 1; n <= this->max_batch_size_; ++n) {
//        LOG(ERROR) << "n " << n << " " << this->layer_types_[i];
//        LOG(ERROR) << "blob_bottom_vec.size() " << blob_bottom_vec.size();
        bottom_batch_size = n;
        this->resize_batch_size(layer_param, bottom_batch_size, filler, blob_bottom_vec);
//        LOG(ERROR) << "blob_top_vec.size() " << blob_top_vec.size();
        top_batch_size = n;
        if (layer_param.type() == "concat") {
          top_batch_size = n * blob_bottom_vec.size();
        }
        this->resize_batch_size(layer_param, top_batch_size, filler, blob_top_vec);
//        LOG(ERROR) << "Forward " << this->layer_types_[i];
        layer->Forward(blob_bottom_vec, &blob_top_vec);
        if (layer_param.type() != "accuracy") {
//          LOG(ERROR) << "Backward " << this->layer_types_[i];
          layer->Backward(blob_top_vec, true, &blob_bottom_vec);
        }
      }  // for (int n = 1; n <= this->max_batch_size_; ++n) {
    }  // for (size_t i = 0; i < this->layer_types_.size(); ++i) {
  }  // for (int n_mode = 0; n_mode < 2; ++n_mode) {
}

}  // namespace caffe
