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
    }

    virtual ~BboxDataLayerTest() {
        delete blob_top_data_;
        delete blob_top_label_;
    }

    int seed_;
    string filename_;
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

    // Iterate through the data
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
    const vector<int> class_idx(arr5, arr5 + sizeof(arr5) / sizeof(arr5[0]));
    for (int iter = 0; iter < 2; ++iter) {
        layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        EXPECT_EQ(this->blob_top_label_->count(), 30);
        EXPECT_EQ(this->blob_top_label_->num(), 30);
        EXPECT_EQ(this->blob_top_label_->data_at(num_objs), 1);
        EXPECT_EQ(this->blob_top_label_->data_at(xmin), 23);
        EXPECT_EQ(this->blob_top_label_->data_at(ymin), 56);
        EXPECT_EQ(this->blob_top_label_->data_at(xmax), 278);
        EXPECT_EQ(this->blob_top_label_->data_at(ymax), 411);
        EXPECT_EQ(this->blob_top_label_->data_at(class_idx), 0);
    }
}


} // namespace caffe
#endif // USE_OPENCV
