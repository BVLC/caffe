#include <cstdio> // remove
#include <string>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

using boost::scoped_ptr;

template <typename TypeParam>
class BigDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BigDataLayerTest()
      : blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        blob_top_ids_(new Blob<Dtype>())
  {}

  virtual void SetUp()
  {
    filename_ = "test/test_data/bigdata.csv";
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    blob_top_vec_.push_back(blob_top_ids_);
    std::cout << "calling setup" << std::endl;
  }


  void TestRead()
  {
    // please update when updating test file
    const size_t cols = 5; 
    const size_t rows = 2;

    Dtype dataSample[] = {-1,-2,1, 2, 0, 1.5, 2.5 ,-1.5, -2.5, 1212.125};
    Dtype labelsSample[] = {1,2};

    // end of manually updated part
    std::cout << "calling testread" << std::endl;

    const size_t Dsize = sizeof(Dtype);
    const size_t MB = 1000000;
    const float  chunk_size = 0.005;
    const size_t batch_size = MB / (cols * Dsize) * chunk_size;
    LayerParameter param;
    param.set_phase(TRAIN);
    ::caffe::BigDataParameter* data_param = param.mutable_big_data_param();
    data_param->set_chunk_size(chunk_size);
    data_param->set_source(filename_.c_str());
    data_param->set_header(1);
    data_param->set_label(6);
    data_param->set_data_start(1);
    data_param->set_data_end(5);
    data_param->set_cache(::caffe::BigDataParameter_CacheControl_DISABLED);

    BigDataLayer<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    // at the begining .. allocate full `chunk_size`MB of space
    EXPECT_EQ( batch_size, blob_top_data_->num() );
    EXPECT_EQ( blob_top_label_->num(), blob_top_data_->num() );
    EXPECT_EQ( 1, blob_top_data_->channels() );
    EXPECT_EQ( 1, blob_top_data_->height() );
    EXPECT_EQ( cols, blob_top_data_->width() );
    EXPECT_EQ( 1, blob_top_label_->channels() );
    EXPECT_EQ( 1, blob_top_label_->height() );
    EXPECT_EQ( 1, blob_top_label_->width() );

    // read once to test if the basic functionality works
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    for (int iter = 0; iter < (rows * 5); ++iter) {
      // test iterations (iter) in order to see the cycling in reading a source file
      EXPECT_EQ(iter % rows, blob_top_ids_->cpu_data()[iter]);
      EXPECT_EQ(labelsSample[iter % rows], blob_top_label_->cpu_data()[iter]);
      for (int j = 0; j < cols; ++j) {
        EXPECT_EQ(dataSample[(iter%rows)*cols+j], blob_top_data_->cpu_data()[(iter*cols)+j]);
      }
    }
    // read again to test cyclability of the source
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // we don't know which ID will appear first - but it must match its data
    int id = blob_top_ids_->cpu_data()[0];
    EXPECT_EQ(labelsSample[id], blob_top_label_->cpu_data()[id]);
    for (int j = 0; j < cols; ++j) {
      EXPECT_EQ(dataSample[id*cols+j], blob_top_data_->cpu_data()[id+j]);
    }
  }

  void TestCache()
  {
    std::cout << "calling testcache" << std::endl;
    LayerParameter param;
    param.set_phase(TRAIN);
    ::caffe::BigDataParameter* data_param = param.mutable_big_data_param();
    data_param->set_chunk_size(0.05);
    data_param->set_source(filename_.c_str());
    data_param->set_header(1);
    data_param->set_label(6);
    data_param->set_data_start(1);
    data_param->set_data_end(5);
    data_param->set_cache(::caffe::BigDataParameter_CacheControl_DISABLED);

    BigDataLayer<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // at this point, a cache file should have been created
    EXPECT_EQ(false, std::ifstream(filename_ + string("bin"), "b").good());
    EXPECT_EQ(false, std::ifstream(filename_ + string("bin.part"), "b").good());
  }

  virtual ~BigDataLayerTest() { 
    std::cout << "calling destructor" << std::endl;
    delete blob_top_data_;
    delete blob_top_label_;
    delete blob_top_ids_;
  }

  string filename_;
  Blob<Dtype>* blob_top_data_;
  Blob<Dtype>* blob_top_label_;
  Blob<Dtype>* blob_top_ids_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BigDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(BigDataLayerTest, TestRead) {
  this->TestRead();
}

// TYPED_TEST(BigDataLayerTest, TestRead) {
//   this->TestCache();
// }

}  // namespace caffe
