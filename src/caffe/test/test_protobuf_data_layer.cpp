#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/util/download_manager.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

#if CV_VERSION_MAJOR == 3
const int CV_IMWRITE_JPEG_QUALITY = cv::IMWRITE_JPEG_QUALITY;
#endif

namespace caffe {

class MockDownloadManager : public DownloadManager {
 public:
  void Download() {
    LOG(INFO) << "MOCK DOWNLOAD";

    cv::Mat image(640, 640, CV_8UC3);
    image = 123;

    for (int row = 120; row < 240; ++row) {
      for (int col = 10; col < 40; ++col) {
        cv::Vec3b& dst = image.at<cv::Vec3b>(row, col);
        dst[0] = row;
        dst[1] = col;
        dst[2] = 200;
      }
    }

    vector<uchar> jpeg;
    vector<int> args;
    args.push_back(CV_IMWRITE_JPEG_QUALITY);
    args.push_back(100);
    CHECK(cv::imencode(".jpg", image, jpeg, args));

    for (vector<string>::const_iterator iter = urls_.begin();
        iter != urls_.end(); ++iter) {
      shared_ptr<stringstream> stream(new stringstream());
      streams_.push_back(stream);

      stream->write(reinterpret_cast<const char*>(jpeg.data()), jpeg.size());
    }
  }
};

DownloadManager* MockDownloadManagerFactory() {
  return new MockDownloadManager();
}

template <typename TypeParam>
class ProtobufDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ProtobufDataLayerTest()
      : blob_top_data_(new Blob<Dtype>()),
        blob_top_single_int_(new Blob<Dtype>()),
        blob_top_single_float_(new Blob<Dtype>()),
        blob_top_single_float_sub_(new Blob<Dtype>()),
        blob_top_single_float_sub_scale_(new Blob<Dtype>()),
        blob_top_weighted_(new Blob<Dtype>()),
        blob_top_multi_int_(new Blob<Dtype>()),
        blob_top_multi_float_(new Blob<Dtype>()),
        blob_top_multi_float_sub_(new Blob<Dtype>()),
        blob_top_multi_float_sub_scale_(new Blob<Dtype>()),
        blob_top_weights_(new Blob<Dtype>()),
        random_test_seed_(1701) {
  }

  virtual void SetUp() {
    filename_.reset(new string());
    MakeTempFilename(filename_.get());
    LOG(INFO) << *filename_;

    blob_top_vec_.push_back(blob_top_data_.get());
    blob_top_vec_.push_back(blob_top_single_int_.get());
    blob_top_vec_.push_back(blob_top_single_float_.get());
    blob_top_vec_.push_back(blob_top_single_float_sub_.get());
    blob_top_vec_.push_back(blob_top_single_float_sub_scale_.get());
    blob_top_vec_.push_back(blob_top_weighted_.get());
    blob_top_vec_.push_back(blob_top_multi_int_.get());
    blob_top_vec_.push_back(blob_top_multi_float_.get());
    blob_top_vec_.push_back(blob_top_multi_float_sub_.get());
    blob_top_vec_.push_back(blob_top_multi_float_sub_scale_.get());
    blob_top_vec_.push_back(blob_top_weights_.get());
  }

  void BuildManifest() {
    ProtobufManifest manifest;

    LabelDefinition* label_def;

    label_def = manifest.add_label_defs();
    label_def->set_name("single_int");
    label_def->set_dim(1);

    label_def = manifest.add_label_defs();
    label_def->set_name("single_float");
    label_def->set_dim(1);

    label_def = manifest.add_label_defs();
    label_def->set_name("single_float_sub");
    label_def->set_dim(1);
    label_def->set_mean(1);

    label_def = manifest.add_label_defs();
    label_def->set_name("single_float_sub_scale");
    label_def->set_dim(1);
    label_def->set_mean(1);
    label_def->set_stdev(2);

    label_def = manifest.add_label_defs();
    label_def->set_name("weighted");
    label_def->set_dim(1);
    label_def->add_weights(1);
    label_def->add_weights(2);
    label_def->add_weights(3);
    label_def->add_weights(4);
    label_def->add_weights(5);

    label_def = manifest.add_label_defs();
    label_def->set_name("multi_int");
    label_def->set_dim(3);

    label_def = manifest.add_label_defs();
    label_def->set_name("multi_float");
    label_def->set_dim(3);

    label_def = manifest.add_label_defs();
    label_def->set_name("multi_float_sub");
    label_def->set_dim(3);
    label_def->set_mean(1);

    label_def = manifest.add_label_defs();
    label_def->set_name("multi_float_sub_scale");
    label_def->set_dim(3);
    label_def->set_mean(1);
    label_def->set_stdev(2);

    for (int i = 0; i < 5; ++i) {
      ProtobufRecord* record = manifest.add_records();

      record->set_image_url("NOT_A_REAL_URL");

      record->set_crop_x1(10);
      record->set_crop_x2(40);

      record->set_crop_y1(120);
      record->set_crop_y2(240);

      TestRecord* test_record = record->MutableExtension(TestRecord::parent);

      test_record->set_single_int(i);
      test_record->set_single_float(i);
      test_record->set_single_float_sub(i);
      test_record->set_single_float_sub_scale(i);
      test_record->set_weighted(i);

      for (int j = 0; j < 3; ++j) {
        test_record->add_multi_int(i + j);
        test_record->add_multi_float(i + j - 1);
        test_record->add_multi_float_sub(i + j - 1);
        test_record->add_multi_float_sub_scale(i + j - 1);
      }
    }

    LOG(INFO) << manifest.DebugString();

    WriteProtoToBinaryFile(manifest, *filename_);
  }

  void TestRead() {
    LayerParameter param;
    ProtobufDataParameter* protobuf_data_param =
        param.mutable_protobuf_data_param();
    protobuf_data_param->set_batch_size(5);
    protobuf_data_param->set_source(filename_->c_str());
    protobuf_data_param->add_labels("single_int");
    protobuf_data_param->add_labels("single_float");
    protobuf_data_param->add_labels("single_float_sub");
    protobuf_data_param->add_labels("single_float_sub_scale");
    protobuf_data_param->add_labels("weighted");
    protobuf_data_param->add_labels("multi_int");
    protobuf_data_param->add_labels("multi_float");
    protobuf_data_param->add_labels("multi_float_sub");
    protobuf_data_param->add_labels("multi_float_sub_scale");
    protobuf_data_param->set_crop_height(4);
    protobuf_data_param->set_crop_width(2);

    ProtobufDataLayer<Dtype> layer(param, MockDownloadManagerFactory);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);

    EXPECT_EQ(5, blob_top_data_->num());
    EXPECT_EQ(3, blob_top_data_->channels());
    EXPECT_EQ(4, blob_top_data_->height());
    EXPECT_EQ(2, blob_top_data_->width());

    EXPECT_EQ(5, blob_top_single_int_->num());
    EXPECT_EQ(1, blob_top_single_int_->channels());
    EXPECT_EQ(1, blob_top_single_int_->height());
    EXPECT_EQ(1, blob_top_single_int_->width());

    EXPECT_EQ(5, blob_top_single_float_->num());
    EXPECT_EQ(1, blob_top_single_float_->channels());
    EXPECT_EQ(1, blob_top_single_float_->height());
    EXPECT_EQ(1, blob_top_single_float_->width());

    EXPECT_EQ(5, blob_top_single_float_sub_->num());
    EXPECT_EQ(1, blob_top_single_float_sub_->channels());
    EXPECT_EQ(1, blob_top_single_float_sub_->height());
    EXPECT_EQ(1, blob_top_single_float_sub_->width());

    EXPECT_EQ(5, blob_top_single_float_sub_scale_->num());
    EXPECT_EQ(1, blob_top_single_float_sub_scale_->channels());
    EXPECT_EQ(1, blob_top_single_float_sub_scale_->height());
    EXPECT_EQ(1, blob_top_single_float_sub_scale_->width());

    EXPECT_EQ(5, blob_top_weighted_->num());
    EXPECT_EQ(1, blob_top_weighted_->channels());
    EXPECT_EQ(1, blob_top_weighted_->height());
    EXPECT_EQ(1, blob_top_weighted_->width());

    EXPECT_EQ(5, blob_top_multi_int_->num());
    EXPECT_EQ(3, blob_top_multi_int_->channels());
    EXPECT_EQ(1, blob_top_multi_int_->height());
    EXPECT_EQ(1, blob_top_multi_int_->width());

    EXPECT_EQ(5, blob_top_multi_float_->num());
    EXPECT_EQ(3, blob_top_multi_float_->channels());
    EXPECT_EQ(1, blob_top_multi_float_->height());
    EXPECT_EQ(1, blob_top_multi_float_->width());

    EXPECT_EQ(5, blob_top_multi_float_sub_->num());
    EXPECT_EQ(3, blob_top_multi_float_sub_->channels());
    EXPECT_EQ(1, blob_top_multi_float_sub_->height());
    EXPECT_EQ(1, blob_top_multi_float_sub_->width());

    EXPECT_EQ(5, blob_top_multi_float_sub_scale_->num());
    EXPECT_EQ(3, blob_top_multi_float_sub_scale_->channels());
    EXPECT_EQ(1, blob_top_multi_float_sub_scale_->height());
    EXPECT_EQ(1, blob_top_multi_float_sub_scale_->width());

    EXPECT_EQ(5, blob_top_weights_->num());
    EXPECT_EQ(1, blob_top_weights_->channels());
    EXPECT_EQ(1, blob_top_weights_->height());
    EXPECT_EQ(1, blob_top_weights_->width());


    for (int iter = 0; iter < 100; ++iter) {
      layer.Forward(blob_bottom_vec_, blob_top_vec_);

      const Dtype* data = blob_top_data_->cpu_data();
      for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(i, blob_top_single_int_->cpu_data()[i]);
        EXPECT_EQ(i, blob_top_single_float_->cpu_data()[i]);
        EXPECT_EQ(i - 1.0, blob_top_single_float_sub_->cpu_data()[i]);
        EXPECT_EQ((i - 1.0) / 2.0,
            blob_top_single_float_sub_scale_->cpu_data()[i]);
        EXPECT_EQ(i, blob_top_weighted_->cpu_data()[i]);
        EXPECT_EQ(i + 1, blob_top_weights_->cpu_data()[i]);

        for (int j = 0; j < 3; ++j) {
          EXPECT_EQ(i + j, blob_top_multi_int_->cpu_data()[3 * i + j])
              << i << " " << j;
          EXPECT_EQ(i + j - 1, blob_top_multi_float_->cpu_data()[3 * i + j])
              << i << " " << j;
          EXPECT_EQ(i + j - 2, blob_top_multi_float_sub_->cpu_data()[3 * i + j])
              << i << " " << j;
          EXPECT_EQ((i + j - 2) / 2.0,
              blob_top_multi_float_sub_scale_->cpu_data()[3 * i + j])
              << i << " " << j;
        }

        for (int channel = 0; channel < 3; ++channel) {
          for (int height = 0; height < 4; ++height) {
            for (int width = 0; width < 2; ++width) {
              Dtype target;
              switch (channel) {
              case 0:
                target = height * Dtype(40) + Dtype(120);
                break;
              case 1:
                target = width * Dtype(30) + Dtype(10);
                break;
              case 2:
                target = Dtype(200);
                break;
              }

              // TODO(kmatzen): Fix this test so that JPEG decompresses closer
              // to the exact value.  In fact, maybe replace jpeg decompressor
              // in the layer with a separate class that I can mock.
              EXPECT_NEAR(target, 128.0 * *data++ + 128.0, 17.0)
                  << channel << " " << height << " " << width;
            }
          }
        }
      }
    }
  }

  shared_ptr<string> filename_;
  shared_ptr<Blob<Dtype> > const blob_top_data_;
  shared_ptr<Blob<Dtype> > const blob_top_single_int_;
  shared_ptr<Blob<Dtype> > const blob_top_single_float_;
  shared_ptr<Blob<Dtype> > const blob_top_single_float_sub_;
  shared_ptr<Blob<Dtype> > const blob_top_single_float_sub_scale_;
  shared_ptr<Blob<Dtype> > const blob_top_weighted_;
  shared_ptr<Blob<Dtype> > const blob_top_multi_int_;
  shared_ptr<Blob<Dtype> > const blob_top_multi_float_;
  shared_ptr<Blob<Dtype> > const blob_top_multi_float_sub_;
  shared_ptr<Blob<Dtype> > const blob_top_multi_float_sub_scale_;
  shared_ptr<Blob<Dtype> > const blob_top_weights_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int random_test_seed_;
};

TYPED_TEST_CASE(ProtobufDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(ProtobufDataLayerTest, TestNothing) {
}

TYPED_TEST(ProtobufDataLayerTest, TestRead) {
  this->BuildManifest();
  this->TestRead();
}

}  // namespace caffe
