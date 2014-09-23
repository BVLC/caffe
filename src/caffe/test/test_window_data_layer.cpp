#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::string;

namespace caffe {

template <typename TypeParam>
class WindowDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  WindowDataLayerTest()
      : blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        blob_data_(new Blob<Dtype>()),
        blob_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
  }
  virtual ~WindowDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
    delete blob_data_;
    delete blob_label_;
  }
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
  void load_window_data_and_label(int batch_size, float fg_fraction,
      bool mirror, int crop_size, string mean_file);
  unsigned int PrefetchRand() {
    CHECK(prefetch_rng_);
    caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    return (*prefetch_rng)();
  }

  shared_ptr<Caffe::RNG> prefetch_rng_;

  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  Blob<Dtype>* const blob_data_;
  Blob<Dtype>* const blob_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename TypeParam>
void WindowDataLayerTest<TypeParam>::load_window_data_and_label(int batch_size,
    float fg_fraction, bool mirror, int crop_size, string mean_file) {
  typedef typename TypeParam::Dtype Dtype;

  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));

  // Setup image and window database;
  vector<string> impath_database;
  impath_database.push_back(EXAMPLES_SOURCE_DIR "images/fish-bike.jpg");
  impath_database.push_back(EXAMPLES_SOURCE_DIR "images/cat.jpg");
  const int fg_num = 1, bg_num = 3;
  const float fg_windows[fg_num][NUM] = {
      {0, 2, 0.700,  10, 100,  70,   150 }};
  const float bg_windows[bg_num][NUM] = {
      {0, 0, 0.100,  50, 130,  100,  140 },
      {1, 0, 0.000,  0,  20,   10,   40  },
      {1, 0, 0.300,  25, 30,   80,   100 }};
  const float fg_labels[fg_num] = {2};
  const float bg_labels[bg_num] = {0, 0, 0};

  // Load mean
  BlobProto blob_proto;
  ReadProtoFromBinaryFile(mean_file, &blob_proto);
  Blob<Dtype> blob_mean;
  blob_mean.FromProto(blob_proto);

  // Load window data
  blob_data_->Reshape(batch_size, 3, crop_size, crop_size);
  blob_label_->Reshape(batch_size, 1, 1, 1);
  Dtype* data = blob_data_->mutable_cpu_data();
  Dtype* label = blob_label_->mutable_cpu_data();
  const Dtype* mean = blob_mean.cpu_data();
  const int mean_off = (blob_mean.width() - crop_size) / 2;
  cv::Size cv_crop_size(crop_size, crop_size);
  // zero out batch
  caffe_set(blob_data_->count(), Dtype(0), data);
  const int num_fg = static_cast<float>(batch_size) * fg_fraction;
  const int num_samples[2] = { batch_size - num_fg, num_fg };
  int item_id = 0;
  // sample from bg set then fg set
  for (int is_fg = 0; is_fg < 2; ++is_fg) {
    for (int dummy = 0; dummy < num_samples[is_fg]; ++dummy) {
      const unsigned int rand_index = PrefetchRand();
      const float* window = (is_fg) ?
          fg_windows[rand_index % fg_num] :
          bg_windows[rand_index % bg_num];
      bool do_mirror = mirror && PrefetchRand() % 2;
      // load the image containing the window
      string impath = impath_database[window[IMAGE_INDEX]];
      cv::Mat im = cv::imread(impath, CV_LOAD_IMAGE_COLOR);
      CHECK(!im.empty()) << "Could not open or find file " << impath;
      const int channels = im.channels();
      int x1 = window[X1], y1 = window[Y1], x2 = window[X2], y2 = window[Y2];
      cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
      cv::Mat im_crop = im(roi);
      cv::resize(im_crop, im_crop, cv_crop_size, 0, 0, cv::INTER_LINEAR);
      if (do_mirror) {
        cv::flip(im_crop, im_crop, 1);
      }
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < im_crop.rows; ++h) {
          for (int w = 0; w < im_crop.cols; ++w) {
            int data_index = blob_data_->offset(item_id, c, h, w);
            int mean_index = blob_mean.offset(0, c, h + mean_off, w + mean_off);
            Dtype pixel = static_cast<Dtype>(im_crop.at<cv::Vec3b>(h, w)[c]);
            data[data_index] = (pixel - mean[mean_index]);
          }
        }
      }
      label[item_id] = (is_fg) ?
          fg_labels[rand_index % fg_num] :
          bg_labels[rand_index % bg_num];
      item_id++;
    }
  }
}

TYPED_TEST_CASE(WindowDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(WindowDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;

  int seed = 1234;
  LayerParameter param;

  // Set up WindowDataParameter
  string source = string(CMAKE_SOURCE_DIR \
      "caffe/test/test_data/sample_window.txt" CMAKE_EXT);
  int batch_size = 4;
  float fg_threshold = 0.5;
  float bg_threshold = 0.5;
  float fg_fraction = 0.25;
  int context_pad = 0;
  string crop_mode = "warp";
  WindowDataParameter* window_data_param = param.mutable_window_data_param();
  window_data_param->set_source(source);
  window_data_param->set_batch_size(batch_size);
  window_data_param->set_fg_threshold(fg_threshold);
  window_data_param->set_bg_threshold(bg_threshold);
  window_data_param->set_fg_fraction(fg_fraction);
  window_data_param->set_context_pad(context_pad);

  // Set up TransformationParameter
  float scale = 1;
  bool mirror = true;
  int crop_size = 10;
  string mean_file = string(CMAKE_SOURCE_DIR \
      "caffe/test/test_data/sample_mean.binaryproto");
  TransformationParameter* transform_param = param.mutable_transform_param();
  transform_param->set_scale(scale);
  transform_param->set_mirror(mirror);
  transform_param->set_crop_size(crop_size);
  transform_param->set_mean_file(mean_file);

  // Test that the layer setup got the correct parameters.
  Caffe::set_random_seed(seed);
  WindowDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), batch_size);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), crop_size);
  EXPECT_EQ(this->blob_top_data_->width(), crop_size);
  EXPECT_EQ(this->blob_top_label_->num(), batch_size);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  // Load window data (and label) and compare them with forward results
  Caffe::set_random_seed(seed);
  this->load_window_data_and_label(batch_size, fg_fraction, mirror,  crop_size,
      mean_file);

  // Compare results
  int data_count = this->blob_data_->count();
  const Dtype* data_top = this->blob_top_data_->cpu_data();
  const Dtype* data = this->blob_data_->cpu_data();
  for (int i = 0; i < data_count; ++i) {
    EXPECT_EQ(data_top[i], data[i]);
  }
  int label_count = this->blob_label_->count();
  const Dtype* label_top = this->blob_top_label_->cpu_data();
  const Dtype* label = this->blob_label_->cpu_data();
  for (int i = 0; i < label_count; ++i) {
    EXPECT_EQ(label_top[i], label[i]);
  }
}

}  // namespace caffe

