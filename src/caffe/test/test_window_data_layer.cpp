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
    source_ = "src/caffe/test/test_data/sample_window.txt";
    batch_size_ = 4;
    fg_threshold_ = 0.5;
    bg_threshold_ = 0.5;
    fg_fraction_ = 0.25;
    context_pad_ = 0;
    crop_mode_ = "warp";
    scale_ = 1;
    mirror_ = true;
    crop_size_ = 10;
    mean_file_ = "src/caffe/test/test_data/sample_mean.binaryproto";
  }
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
  void load_window_data();
  unsigned int PrefetchRand() {
    CHECK(prefetch_rng_);
    caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    return (*prefetch_rng)();
  }

  virtual ~WindowDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
    delete blob_data_;
    delete blob_label_;
  }

  string source_;
  int batch_size_;
  float fg_threshold_;
  float bg_threshold_;
  float fg_fraction_;
  int context_pad_;
  string crop_mode_;
  float scale_;
  bool mirror_;
  int crop_size_;
  string mean_file_;

  shared_ptr<Caffe::RNG> prefetch_rng_;

  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  Blob<Dtype>* const blob_data_;
  Blob<Dtype>* const blob_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename TypeParam>
void WindowDataLayerTest<TypeParam>::load_window_data() {
  typedef typename TypeParam::Dtype Dtype;

  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));

  // Load mean
  BlobProto blob_proto;
  ReadProtoFromBinaryFile(mean_file_, &blob_proto);
  Blob<Dtype> blob_mean;
  blob_mean.FromProto(blob_proto);

  // Load window file
  vector<vector<float> > fg_windows, bg_windows;
  vector<std::pair<std::string, vector<int> > > image_database;
  std::ifstream infile(source_.c_str());
  CHECK(infile.good()) << "Failed to open window file " << source_;
  string hashtag;
  int image_index;
  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile >> image_path;
    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    image_database.push_back(std::make_pair(image_path, image_size));
    // read each box
    int num_windows;
    infile >> num_windows;
    for (int i = 0; i < num_windows; ++i) {
      int label, x1, y1, x2, y2;
      float overlap;
      infile >> label >> overlap >> x1 >> y1 >> x2 >> y2;
      vector<float> window(NUM);
      window[IMAGE_INDEX] = image_index;
      window[LABEL] = label;
      window[OVERLAP] = overlap;
      window[X1] = x1;
      window[Y1] = y1;
      window[X2] = x2;
      window[Y2] = y2;
      // add window to foreground list or background list
      if (overlap >= fg_threshold_) {
        int label = window[LABEL];
        CHECK_GT(label, 0);
        fg_windows.push_back(window);
      } else if (overlap < bg_threshold_) {
        // background window, force label and overlap to 0
        window[LABEL] = 0;
        window[OVERLAP] = 0;
        bg_windows.push_back(window);
      }
    }
  } while (infile >> hashtag >> image_index);

  // Load window data
  blob_data_->Reshape(batch_size_, 3, crop_size_, crop_size_);
  blob_label_->Reshape(batch_size_, 1, 1, 1);
  Dtype* top_data = blob_data_->mutable_cpu_data();
  Dtype* top_label = blob_label_->mutable_cpu_data();
  const Dtype* mean = blob_mean.cpu_data();
  const int mean_off = (blob_mean.width() - crop_size_) / 2;
  const int mean_width = blob_mean.width();
  const int mean_height = blob_mean.height();
  cv::Size cv_crop_size(crop_size_, crop_size_);
  bool use_square = (crop_mode_ == "square") ? true : false;

  // zero out batch
  caffe_set(blob_data_->count(), Dtype(0), top_data);

  const int num_fg = static_cast<int>(static_cast<float>(batch_size_)
      * fg_fraction_);
  const int num_samples[2] = { batch_size_ - num_fg, num_fg };
  int item_id = 0;
  // sample from bg set then fg set
  for (int is_fg = 0; is_fg < 2; ++is_fg) {
    for (int dummy = 0; dummy < num_samples[is_fg]; ++dummy) {
      // sample a window
      const unsigned int rand_index = PrefetchRand();
      vector<float> window = (is_fg) ?
          fg_windows[rand_index % fg_windows.size()] :
          bg_windows[rand_index % bg_windows.size()];
      bool do_mirror = false;
      if (mirror_ && PrefetchRand() % 2) {
        do_mirror = true;
      }
      // load the image containing the window
      pair<std::string, vector<int> > image =
          image_database[window[IMAGE_INDEX]];

      cv::Mat cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
      if (!cv_img.data) {
        LOG(ERROR) << "Could not open or find file " << image.first;
        return;
      }
      const int channels = cv_img.channels();

      // crop window out of image and warp it
      int x1 = window[X1];
      int y1 = window[Y1];
      int x2 = window[X2];
      int y2 = window[Y2];
      int pad_w = 0;
      int pad_h = 0;
      if (context_pad_ > 0 || use_square) {
        // scale factor by which to expand the original region
        // such that after warping the expanded region to crop_size x crop_size
        // there's exactly context_pad amount of padding on each side
        Dtype context_scale = static_cast<Dtype>(crop_size_) /
            static_cast<Dtype>(crop_size_ - 2*context_pad_);

        // compute the expanded region
        Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
        Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
        Dtype center_x = static_cast<Dtype>(x1) + half_width;
        Dtype center_y = static_cast<Dtype>(y1) + half_height;
        if (use_square) {
          if (half_height > half_width) {
            half_width = half_height;
          } else {
            half_height = half_width;
          }
        }
        x1 = static_cast<int>(round(center_x - half_width*context_scale));
        x2 = static_cast<int>(round(center_x + half_width*context_scale));
        y1 = static_cast<int>(round(center_y - half_height*context_scale));
        y2 = static_cast<int>(round(center_y + half_height*context_scale));

        // the expanded region may go outside of the image
        // so we compute the clipped (expanded) region and keep track of
        // the extent beyond the image
        int unclipped_height = y2-y1+1;
        int unclipped_width = x2-x1+1;
        int pad_x1 = std::max(0, -x1);
        int pad_y1 = std::max(0, -y1);
        int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
        int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
        // clip bounds
        x1 = x1 + pad_x1;
        x2 = x2 - pad_x2;
        y1 = y1 + pad_y1;
        y2 = y2 - pad_y2;
        CHECK_GT(x1, -1);
        CHECK_GT(y1, -1);
        CHECK_LT(x2, cv_img.cols);
        CHECK_LT(y2, cv_img.rows);

        int clipped_height = y2-y1+1;
        int clipped_width = x2-x1+1;

        // scale factors that would be used to warp the unclipped
        // expanded region
        Dtype scale_x =
            static_cast<Dtype>(crop_size_)/static_cast<Dtype>(unclipped_width);
        Dtype scale_y =
            static_cast<Dtype>(crop_size_)/static_cast<Dtype>(unclipped_height);

        // size to warp the clipped expanded region to
        cv_crop_size.width =
            static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
        cv_crop_size.height =
            static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
        pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
        pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
        pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
        pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));

        pad_h = pad_y1;
        // if we're mirroring, we mirror the padding too (to be pedantic)
        if (do_mirror) {
          pad_w = pad_x2;
        } else {
          pad_w = pad_x1;
        }
        // ensure that the warped, clipped region plus the padding fits in the
        // crop_size x crop_size image (it might not due to rounding)
        if (pad_h + cv_crop_size.height > crop_size_) {
          cv_crop_size.height = crop_size_ - pad_h;
        }
        if (pad_w + cv_crop_size.width > crop_size_) {
          cv_crop_size.width = crop_size_ - pad_w;
        }
      }
      cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
      cv::Mat cv_cropped_img = cv_img(roi);
      cv::resize(cv_cropped_img, cv_cropped_img,
          cv_crop_size, 0, 0, cv::INTER_LINEAR);
      // horizontal flip at random
      if (do_mirror) {
        cv::flip(cv_cropped_img, cv_cropped_img, 1);
      }
      // copy the warped window into top_data
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < cv_cropped_img.rows; ++h) {
          for (int w = 0; w < cv_cropped_img.cols; ++w) {
            Dtype pixel =
                static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);
            top_data[((item_id * channels + c) * crop_size_ + h + pad_h)
                     * crop_size_ + w + pad_w]
                = (pixel
                    - mean[(c * mean_height + h + mean_off + pad_h)
                           * mean_width + w + mean_off + pad_w])
                  * scale_;
          }
        }
      }

      // get window label
      top_label[item_id] = window[LABEL];
      item_id++;
    }
  }
}

TYPED_TEST_CASE(WindowDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(WindowDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter param;
  WindowDataParameter* window_data_param = param.mutable_window_data_param();
  window_data_param->set_source(this->source_);
  window_data_param->set_batch_size(this->batch_size_);
  window_data_param->set_fg_threshold(this->fg_threshold_);
  window_data_param->set_bg_threshold(this->bg_threshold_);
  window_data_param->set_fg_fraction(this->fg_fraction_);
  window_data_param->set_context_pad(this->context_pad_);
  TransformationParameter* transform_param = param.mutable_transform_param();
  transform_param->set_scale(this->scale_);
  transform_param->set_mirror(this->mirror_);
  transform_param->set_crop_size(this->crop_size_);
  transform_param->set_mean_file(this->mean_file_);

  // Test that the layer setup got the correct parameters.
  int seed = 1234;
  Caffe::set_random_seed(seed);
  WindowDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), this->batch_size_);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), this->crop_size_);
  EXPECT_EQ(this->blob_top_data_->width(), this->crop_size_);

  EXPECT_EQ(this->blob_top_label_->num(), this->batch_size_);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
  Caffe::set_random_seed(seed);

  this->load_window_data();

  // Compare results
  int data_count = this->blob_data_->count();
  const Dtype* data_top = this->blob_top_data_->cpu_data();
  const Dtype* data = this->blob_data_->cpu_data();
  for (int i = 0; i < data_count; ++i) {
    CHECK_EQ(data_top[i], data[i]);
  }
  int label_count = this->blob_label_->count();
  const Dtype* label_top = this->blob_top_label_->cpu_data();
  const Dtype* label = this->blob_label_->cpu_data();
  for (int i = 0; i < label_count; ++i) {
    CHECK_EQ(label_top[i], label[i]);
  }
}

}  // namespace caffe

