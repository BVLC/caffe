#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class ImageDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ImageDataLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_data_2_(new Blob<Dtype>()),
        blob_top_data_3_(new Blob<Dtype>()),
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
      outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg " << i << "\n";
    }
    outfile.close();
    // Create test input file for images of distinct sizes.
    MakeTempFilename(&filename_reshape_);
    std::ofstream reshapefile(filename_reshape_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_reshape_;
    reshapefile << EXAMPLES_SOURCE_DIR "images/cat.jpg " << 0 << "\n";
    reshapefile << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg " << 1 << "\n";
    reshapefile.close();

    // Create test input files for multiple images as an input (siamese etc.).
    MakeTempFilename(&filename_siamese_);
    std::ofstream siamesefile(filename_siamese_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_siamese_;
    siamesefile << EXAMPLES_SOURCE_DIR "images/cat_1.jpg "
                << EXAMPLES_SOURCE_DIR "images/cat_1.jpg "
                << 1 << "\n";
    siamesefile << EXAMPLES_SOURCE_DIR "images/cat_1.jpg "
                << EXAMPLES_SOURCE_DIR "images/cat_2.jpg "
                << 0 << "\n";
    siamesefile << EXAMPLES_SOURCE_DIR "images/cat_2.jpg "
                << EXAMPLES_SOURCE_DIR "images/cat_2.jpg "
                << 1 << "\n";
    siamesefile.close();

    MakeTempFilename(&filename_tripple_);
    std::ofstream tripplefile(filename_tripple_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_tripple_;
    tripplefile << EXAMPLES_SOURCE_DIR "images/cat_1.jpg "
                << EXAMPLES_SOURCE_DIR "images/cat_2.jpg "
                << EXAMPLES_SOURCE_DIR "images/cat_3.jpg "
                << 0 << "\n";
    tripplefile << EXAMPLES_SOURCE_DIR "images/cat_1.jpg "
                << EXAMPLES_SOURCE_DIR "images/cat_1.jpg "
                << EXAMPLES_SOURCE_DIR "images/cat_2.jpg "
                << 1 << "\n";
    tripplefile << EXAMPLES_SOURCE_DIR "images/cat_1.jpg "
                << EXAMPLES_SOURCE_DIR "images/cat_2.jpg "
                << EXAMPLES_SOURCE_DIR "images/cat_1.jpg "
                << 2 << "\n";
    tripplefile << EXAMPLES_SOURCE_DIR "images/cat_2.jpg "
                << EXAMPLES_SOURCE_DIR "images/cat_1.jpg "
                << EXAMPLES_SOURCE_DIR "images/cat_1.jpg "
                << 3 << "\n";
    tripplefile << EXAMPLES_SOURCE_DIR "images/cat_1.jpg "
                << EXAMPLES_SOURCE_DIR "images/cat_1.jpg "
                << EXAMPLES_SOURCE_DIR "images/cat_1.jpg "
                << 4 << "\n";
    tripplefile.close();
  }

  virtual ~ImageDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_data_2_;
    delete blob_top_data_3_;
    delete blob_top_label_;
  }

  int seed_;
  string filename_;
  string filename_reshape_;
  string filename_siamese_;
  string filename_tripple_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_data_2_;
  Blob<Dtype>* const blob_top_data_3_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ImageDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(ImageDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  param.add_top("img");
  param.add_top("label");
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestResize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  param.add_top("img");
  param.add_top("label");
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_new_height(256);
  image_data_param->set_new_width(256);
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 256);
  EXPECT_EQ(this->blob_top_data_->width(), 256);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  param.add_top("img");
  param.add_top("label");
  image_data_param->set_batch_size(1);
  image_data_param->set_source(this->filename_reshape_.c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_label_->num(), 1);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // cat.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  // fish-bike.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 323);
  EXPECT_EQ(this->blob_top_data_->width(), 481);
}

TYPED_TEST(ImageDataLayerTest, TestShuffle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  param.add_top("img");
  param.add_top("label");
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(true);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    map<Dtype, int> values_to_indices;
    int num_in_order = 0;
    for (int i = 0; i < 5; ++i) {
      Dtype value = this->blob_top_label_->cpu_data()[i];
      // Check that the value has not been seen already (no duplicates).
      EXPECT_EQ(values_to_indices.find(value), values_to_indices.end());
      values_to_indices[value] = i;
      num_in_order += (value == Dtype(i));
    }
    EXPECT_EQ(5, values_to_indices.size());
    EXPECT_GT(5, num_in_order);
  }
}

TYPED_TEST(ImageDataLayerTest, TestSiamese) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->blob_top_data_);
  this->blob_top_vec_.push_back(this->blob_top_data_2_);
  this->blob_top_vec_.push_back(this->blob_top_label_);
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  param.add_top("img1");
  param.add_top("img2");
  param.add_top("label");
  image_data_param->set_batch_size(3);
  image_data_param->set_source(this->filename_siamese_.c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->num(), 3);
  EXPECT_EQ(this->blob_top_vec_[0]->channels(), 3);
  EXPECT_EQ(this->blob_top_vec_[0]->height(), 360);
  EXPECT_EQ(this->blob_top_vec_[0]->width(), 480);
  EXPECT_EQ(this->blob_top_vec_[1]->num(), 3);
  EXPECT_EQ(this->blob_top_vec_[1]->channels(), 3);
  EXPECT_EQ(this->blob_top_vec_[1]->height(), 360);
  EXPECT_EQ(this->blob_top_vec_[1]->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 3);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(1, this->blob_top_label_->cpu_data()[0]);
    EXPECT_EQ(0, this->blob_top_label_->cpu_data()[1]);
    EXPECT_EQ(1, this->blob_top_label_->cpu_data()[2]);
    int n = 0;
    for (int c = 0; c < 3; c++) {
      for (int h = 0; h < 360; h++) {
        for (int w = 0; w < 480; w++) {
          EXPECT_EQ(this->blob_top_data_->data_at(n, c, h, w),
                    this->blob_top_data_2_->data_at(n, c, h, w));
        }
      }
    }
    n = 2;
    for (int c = 0; c < 3; c++) {
      for (int h = 0; h < 360; h++) {
        for (int w = 0; w < 480; w++) {
          EXPECT_EQ(this->blob_top_data_->data_at(n, c, h, w),
                    this->blob_top_data_2_->data_at(n, c, h, w));
        }
      }
    }
  }
}


TYPED_TEST(ImageDataLayerTest, TestTripple) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->blob_top_data_);
  this->blob_top_vec_.push_back(this->blob_top_data_2_);
  this->blob_top_vec_.push_back(this->blob_top_data_3_);
  this->blob_top_vec_.push_back(this->blob_top_label_);
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  param.add_top("img1");
  param.add_top("img2");
  param.add_top("img3");
  param.add_top("label");
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_tripple_.c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->num(), 5);
  EXPECT_EQ(this->blob_top_vec_[0]->channels(), 3);
  EXPECT_EQ(this->blob_top_vec_[0]->height(), 360);
  EXPECT_EQ(this->blob_top_vec_[0]->width(), 480);
  EXPECT_EQ(this->blob_top_vec_[1]->num(), 5);
  EXPECT_EQ(this->blob_top_vec_[1]->channels(), 3);
  EXPECT_EQ(this->blob_top_vec_[1]->height(), 360);
  EXPECT_EQ(this->blob_top_vec_[1]->width(), 480);
  EXPECT_EQ(this->blob_top_vec_[2]->num(), 5);
  EXPECT_EQ(this->blob_top_vec_[2]->channels(), 3);
  EXPECT_EQ(this->blob_top_vec_[2]->height(), 360);
  EXPECT_EQ(this->blob_top_vec_[2]->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
    for (int n = 0; n < 5; n++) {
      for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 360; h++) {
          for (int w = 0; w < 480; w++) {
            if (n == 1 || n == 4) {
              EXPECT_EQ(this->blob_top_data_->data_at(n, c, h, w),
                        this->blob_top_data_2_->data_at(n, c, h, w));
            }
            if (n == 2 || n == 4) {
              EXPECT_EQ(this->blob_top_data_->data_at(n, c, h, w),
                        this->blob_top_data_3_->data_at(n, c, h, w));
            }
            if (n == 3 || n == 4) {
              EXPECT_EQ(this->blob_top_data_3_->data_at(n, c, h, w),
                        this->blob_top_data_2_->data_at(n, c, h, w));
            }
          }
        }
      }
    }
  }
}
}  // namespace caffe
