#ifdef USE_OPENCV
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/bbox_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class BboxDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BboxDataLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create test input file.
    MakeTempFilename(&filename_);
    std::ofstream outfile(filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_;
    for (int i = 0; i < 5; ++i) {
      outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg" << " "
              << EXAMPLES_SOURCE_DIR "annotations/cat.txt"
              << std::endl;
    }
    outfile.close();
    // Multiple objects in an image
    MakeTempFilename(&filename_multiobj_);
    std::ofstream multifile(filename_multiobj_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_multiobj_;
    multifile << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg" << " "
              << EXAMPLES_SOURCE_DIR "annotations/fish-bike.txt"
              << std::endl;
    multifile.close();
    // Multi image and multi boxes
    MakeTempFilename(&filename_multi_multi);
    std::ofstream multi_multi(filename_multi_multi.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_multi_multi;
    multi_multi << EXAMPLES_SOURCE_DIR "images/cat.jpg" << " "
                << EXAMPLES_SOURCE_DIR "annotations/cat.txt"
                << std::endl
                << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg" << " "
                << EXAMPLES_SOURCE_DIR "annotations/fish-bike.txt"
                << std::endl;
    multi_multi.close();
  }

  virtual ~BboxDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  int seed_;
  string filename_;
  string filename_multiobj_;
  string filename_multi_multi;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BboxDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(BboxDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  BboxDataParameter* bbox_data_param = param.mutable_bbox_data_param();
  bbox_data_param->set_batch_size(5);
  bbox_data_param->set_source(this->filename_.c_str());
  bbox_data_param->set_shuffle(false);
  BboxDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 6);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  const int arr0[] = {0, 0, 0, 0};
  const vector<int> num_objs(arr0, arr0 + sizeof(arr0) / sizeof(arr0[0]));
  const int arr1[] = {1, 0, 0, 0};
  const vector<int> xmin(arr1, arr1 + sizeof(arr1) / sizeof(arr1[0]));
  const int arr2[] = {2, 0, 0, 0};
  const vector<int> ymin(arr2, arr2 + sizeof(arr2) / sizeof(arr2[0]));
  const int arr3[] = {3, 0, 0, 0};
  const vector<int> xmax(arr3, arr3 + sizeof(arr3) / sizeof(arr3[0]));
  const int arr4[] = {4, 0, 0, 0};
  const vector<int> ymax(arr4, arr4 + sizeof(arr4) / sizeof(arr4[0]));
  const int arr5[] = {5, 0, 0, 0};
  const vector<int> class_idx1(arr5, arr5 + sizeof(arr5) / sizeof(arr5[0]));
  const int arr6[] = {6, 0, 0, 0};
  const vector<int> num_objs2(arr6, arr6 + sizeof(arr6) / sizeof(arr6[0]));
  const int arr7[] = {7, 0, 0, 0};
  const vector<int> xmin2(arr7, arr7 + sizeof(arr7) / sizeof(arr7[0]));
  // Iterate through the data
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_label_->count(), 30);
    EXPECT_EQ(this->blob_top_label_->num(), 30);
    EXPECT_EQ(this->blob_top_label_->data_at(num_objs), 1);
    EXPECT_EQ(this->blob_top_label_->data_at(xmin), 23);
    EXPECT_EQ(this->blob_top_label_->data_at(ymin), 56);
    EXPECT_EQ(this->blob_top_label_->data_at(xmax), 278);
    EXPECT_EQ(this->blob_top_label_->data_at(ymax), 411);
    EXPECT_EQ(this->blob_top_label_->data_at(class_idx1), 0);
    EXPECT_EQ(this->blob_top_label_->data_at(num_objs2), 1);
    EXPECT_EQ(this->blob_top_label_->data_at(xmin2), 23);
  }
}

TYPED_TEST(BboxDataLayerTest, TestMulti) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter param;
    BboxDataParameter* bbox_data_param = param.mutable_bbox_data_param();
    bbox_data_param->set_batch_size(1);
    bbox_data_param->set_source(this->filename_multiobj_.c_str());
    bbox_data_param->set_shuffle(false);
    BboxDataLayer<Dtype> layer(param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_data_->num(), 1);
    EXPECT_EQ(this->blob_top_data_->channels(), 3);
    EXPECT_EQ(this->blob_top_data_->height(), 323);
    EXPECT_EQ(this->blob_top_data_->width(), 481);
    EXPECT_EQ(this->blob_top_label_->num(), 11);
    EXPECT_EQ(this->blob_top_label_->channels(), 1);
    EXPECT_EQ(this->blob_top_label_->width(), 1);
    EXPECT_EQ(this->blob_top_label_->width(), 1);
    // bbox related annotations
    EXPECT_EQ(this->blob_top_label_->count(), 11);
    EXPECT_EQ(this->blob_top_label_->data_at(0, 0, 0, 0), 2);
    EXPECT_EQ(this->blob_top_label_->data_at(1, 0, 0, 0), 22);
    EXPECT_EQ(this->blob_top_label_->data_at(2, 0, 0, 0), 45);
    EXPECT_EQ(this->blob_top_label_->data_at(3, 0, 0, 0), 153);
    EXPECT_EQ(this->blob_top_label_->data_at(4, 0, 0, 0), 221);
    EXPECT_EQ(this->blob_top_label_->data_at(5, 0, 0, 0), 21);
    EXPECT_EQ(this->blob_top_label_->data_at(6, 0, 0, 0), 32);
    EXPECT_EQ(this->blob_top_label_->data_at(7, 0, 0, 0), 155);
    EXPECT_EQ(this->blob_top_label_->data_at(8, 0, 0, 0), 421);
    EXPECT_EQ(this->blob_top_label_->data_at(9, 0, 0, 0), 301);
    EXPECT_EQ(this->blob_top_label_->data_at(10, 0, 0, 0), 22);
}

TYPED_TEST(BboxDataLayerTest, TestMultiMulti) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter param;
    BboxDataParameter* bbox_data_param = param.mutable_bbox_data_param();
    bbox_data_param->set_batch_size(2);
    bbox_data_param->set_source(this->filename_multi_multi.c_str());
    bbox_data_param->set_shuffle(false);
    bbox_data_param->set_new_height(416);
    bbox_data_param->set_new_width(416);
    BboxDataLayer<Dtype> layer(param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    EXPECT_EQ(this->blob_top_data_->num(), 2);
    EXPECT_EQ(this->blob_top_data_->channels(), 3);
    EXPECT_EQ(this->blob_top_data_->height(), 416);
    EXPECT_EQ(this->blob_top_data_->width(), 416);
    EXPECT_EQ(this->blob_top_label_->count(), 17);
    // cat.jpg
    EXPECT_EQ(this->blob_top_label_->data_at(0, 0, 0, 0), 1);
    EXPECT_EQ(this->blob_top_label_->data_at(1, 0, 0, 0), 23);
    EXPECT_EQ(this->blob_top_label_->data_at(2, 0, 0, 0), 56);
    EXPECT_EQ(this->blob_top_label_->data_at(3, 0, 0, 0), 278);
    EXPECT_EQ(this->blob_top_label_->data_at(4, 0, 0, 0), 411);
    EXPECT_EQ(this->blob_top_label_->data_at(5, 0, 0, 0), 0);
    // fish-bike.jpg
    EXPECT_EQ(this->blob_top_label_->data_at(6, 0, 0, 0), 2);
    EXPECT_EQ(this->blob_top_label_->data_at(7, 0, 0, 0), 22);
    EXPECT_EQ(this->blob_top_label_->data_at(8, 0, 0, 0), 45);
    EXPECT_EQ(this->blob_top_label_->data_at(9, 0, 0, 0), 153);
    EXPECT_EQ(this->blob_top_label_->data_at(10, 0, 0, 0), 221);
    EXPECT_EQ(this->blob_top_label_->data_at(11, 0, 0, 0), 21);
    EXPECT_EQ(this->blob_top_label_->data_at(12, 0, 0, 0), 32);
    EXPECT_EQ(this->blob_top_label_->data_at(13, 0, 0, 0), 155);
    EXPECT_EQ(this->blob_top_label_->data_at(14, 0, 0, 0), 421);
    EXPECT_EQ(this->blob_top_label_->data_at(15, 0, 0, 0), 301);
    EXPECT_EQ(this->blob_top_label_->data_at(16, 0, 0, 0), 22);
}

}  // namespace caffe
#endif  // USE_OPENCV
