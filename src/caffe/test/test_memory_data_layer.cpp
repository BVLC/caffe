#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/memory_data_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class MemoryDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MemoryDataLayerTest()
    : data_(new Blob<Dtype>()),
      labels_(new Blob<Dtype>()),
      data_blob_(new Blob<Dtype>()),
      label_blob_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    batch_size_ = 8;
    batches_ = 12;
    channels_ = 4;
    height_ = 7;
    width_ = 11;
    blob_top_vec_.push_back(data_blob_);
    blob_top_vec_.push_back(label_blob_);
    // pick random input data
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    data_->Reshape(batches_ * batch_size_, channels_, height_, width_);
    labels_->Reshape(batches_ * batch_size_, 1, 1, 1);
    filler.Fill(this->data_);
    filler.Fill(this->labels_);
  }

  virtual ~MemoryDataLayerTest() {
    delete data_blob_;
    delete label_blob_;
    delete data_;
    delete labels_;
  }
  int batch_size_;
  int batches_;
  int channels_;
  int height_;
  int width_;
  // we don't really need blobs for the input data, but it makes it
  //  easier to call Filler
  Blob<Dtype>* const data_;
  Blob<Dtype>* const labels_;
  // blobs for the top of MemoryDataLayer
  Blob<Dtype>* const data_blob_;
  Blob<Dtype>* const label_blob_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MemoryDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(MemoryDataLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  MemoryDataParameter* md_param = layer_param.mutable_memory_data_param();
  md_param->set_batch_size(this->batch_size_);
  md_param->set_channels(this->channels_);
  md_param->set_height(this->height_);
  md_param->set_width(this->width_);
  shared_ptr<Layer<Dtype> > layer(
      new MemoryDataLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->data_blob_->num(), this->batch_size_);
  EXPECT_EQ(this->data_blob_->channels(), this->channels_);
  EXPECT_EQ(this->data_blob_->height(), this->height_);
  EXPECT_EQ(this->data_blob_->width(), this->width_);
  EXPECT_EQ(this->label_blob_->num(), this->batch_size_);
  EXPECT_EQ(this->label_blob_->channels(), 1);
  EXPECT_EQ(this->label_blob_->height(), 1);
  EXPECT_EQ(this->label_blob_->width(), 1);
}

// run through a few batches and check that the right data appears
TYPED_TEST(MemoryDataLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  MemoryDataParameter* md_param = layer_param.mutable_memory_data_param();
  md_param->set_batch_size(this->batch_size_);
  md_param->set_channels(this->channels_);
  md_param->set_height(this->height_);
  md_param->set_width(this->width_);
  shared_ptr<MemoryDataLayer<Dtype> > layer(
      new MemoryDataLayer<Dtype>(layer_param));
  layer->DataLayerSetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Reset(this->data_->mutable_cpu_data(),
      this->labels_->mutable_cpu_data(), this->data_->num());
  for (int i = 0; i < this->batches_ * 6; ++i) {
    int batch_num = i % this->batches_;
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int j = 0; j < this->data_blob_->count(); ++j) {
      EXPECT_EQ(this->data_blob_->cpu_data()[j],
          this->data_->cpu_data()[
              this->data_->offset(1) * this->batch_size_ * batch_num + j]);
    }
    for (int j = 0; j < this->label_blob_->count(); ++j) {
      EXPECT_EQ(this->label_blob_->cpu_data()[j],
          this->labels_->cpu_data()[this->batch_size_ * batch_num + j]);
    }
  }
}

#ifdef USE_OPENCV
TYPED_TEST(MemoryDataLayerTest, AddDatumVectorDefaultTransform) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter param;
  MemoryDataParameter* memory_data_param = param.mutable_memory_data_param();
  memory_data_param->set_batch_size(this->batch_size_);
  memory_data_param->set_channels(this->channels_);
  memory_data_param->set_height(this->height_);
  memory_data_param->set_width(this->width_);
  MemoryDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // We add batch_size*num_iter items, then for each iteration
  // we forward batch_size elements
  int num_iter = 5;
  vector<Datum> datum_vector(this->batch_size_ * num_iter);
  const size_t count = this->channels_ * this->height_ * this->width_;
  size_t pixel_index = 0;
  for (int i = 0; i < this->batch_size_ * num_iter; ++i) {
    datum_vector[i].set_channels(this->channels_);
    datum_vector[i].set_height(this->height_);
    datum_vector[i].set_width(this->width_);
    datum_vector[i].set_label(i);
    vector<char> pixels(count);
    for (int j = 0; j < count; ++j) {
      pixels[j] = pixel_index++ % 256;
    }
    datum_vector[i].set_data(&(pixels[0]), count);
  }
  layer.AddDatumVector(datum_vector);

  int data_index;
  // Go through the data 5 times
  for (int iter = 0; iter < num_iter; ++iter) {
    int offset = this->batch_size_ * iter;
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->data_blob_->cpu_data();
    size_t index = 0;
    for (int i = 0; i < this->batch_size_; ++i) {
      const string& data_string = datum_vector[offset + i].data();
      EXPECT_EQ(offset + i, this->label_blob_->cpu_data()[i]);
      for (int c = 0; c < this->channels_; ++c) {
        for (int h = 0; h < this->height_; ++h) {
          for (int w = 0; w < this->width_; ++w) {
            data_index = (c * this->height_ + h) * this->width_ + w;
            EXPECT_EQ(static_cast<Dtype>(
                static_cast<uint8_t>(data_string[data_index])),
                      data[index++]);
          }
        }
      }
    }
  }
}

TYPED_TEST(MemoryDataLayerTest, AddMatVectorDefaultTransform) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  MemoryDataParameter* memory_data_param = param.mutable_memory_data_param();
  memory_data_param->set_batch_size(this->batch_size_);
  memory_data_param->set_channels(this->channels_);
  memory_data_param->set_height(this->height_);
  memory_data_param->set_width(this->width_);
  MemoryDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // We add batch_size*num_iter items, then for each iteration
  // we forward batch_size elements
  int num_iter = 5;
  vector<cv::Mat> mat_vector(this->batch_size_ * num_iter);
  vector<int> label_vector(this->batch_size_ * num_iter);
  for (int i = 0; i < this->batch_size_*num_iter; ++i) {
    mat_vector[i] = cv::Mat(this->height_, this->width_, CV_8UC4);
    label_vector[i] = i;
    cv::randu(mat_vector[i], cv::Scalar::all(0), cv::Scalar::all(255));
  }
  layer.AddMatVector(mat_vector, label_vector);

  int data_index;
  const size_t count = this->channels_ * this->height_ * this->width_;
  for (int iter = 0; iter < num_iter; ++iter) {
    int offset = this->batch_size_ * iter;
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->data_blob_->cpu_data();
    for (int i = 0; i < this->batch_size_; ++i) {
      EXPECT_EQ(offset + i, this->label_blob_->cpu_data()[i]);
      for (int h = 0; h < this->height_; ++h) {
        const unsigned char* ptr_mat = mat_vector[offset + i].ptr<uchar>(h);
        int index = 0;
        for (int w = 0; w < this->width_; ++w) {
          for (int c = 0; c < this->channels_; ++c) {
            data_index = (i*count) + (c * this->height_ + h) * this->width_ + w;
            Dtype pixel = static_cast<Dtype>(ptr_mat[index++]);
            EXPECT_EQ(static_cast<int>(pixel),
                      data[data_index]);
          }
        }
      }
    }
  }
}

TYPED_TEST(MemoryDataLayerTest, TestSetBatchSize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  MemoryDataParameter* memory_data_param = param.mutable_memory_data_param();
  memory_data_param->set_batch_size(this->batch_size_);
  memory_data_param->set_channels(this->channels_);
  memory_data_param->set_height(this->height_);
  memory_data_param->set_width(this->width_);
  MemoryDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // first add data as usual
  int num_iter = 5;
  vector<cv::Mat> mat_vector(this->batch_size_ * num_iter);
  vector<int> label_vector(this->batch_size_ * num_iter);
  for (int i = 0; i < this->batch_size_*num_iter; ++i) {
    mat_vector[i] = cv::Mat(this->height_, this->width_, CV_8UC4);
    label_vector[i] = i;
    cv::randu(mat_vector[i], cv::Scalar::all(0), cv::Scalar::all(255));
  }
  layer.AddMatVector(mat_vector, label_vector);
  // then consume the data
  int data_index;
  const size_t count = this->channels_ * this->height_ * this->width_;
  for (int iter = 0; iter < num_iter; ++iter) {
    int offset = this->batch_size_ * iter;
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->data_blob_->cpu_data();
    for (int i = 0; i < this->batch_size_; ++i) {
      EXPECT_EQ(offset + i, this->label_blob_->cpu_data()[i]);
      for (int h = 0; h < this->height_; ++h) {
        const unsigned char* ptr_mat = mat_vector[offset + i].ptr<uchar>(h);
        int index = 0;
        for (int w = 0; w < this->width_; ++w) {
          for (int c = 0; c < this->channels_; ++c) {
            data_index = (i*count) + (c * this->height_ + h) * this->width_ + w;
            Dtype pixel = static_cast<Dtype>(ptr_mat[index++]);
            EXPECT_EQ(static_cast<int>(pixel), data[data_index]);
          }
        }
      }
    }
  }
  // and then add new data with different batch_size
  int new_batch_size = 16;
  layer.set_batch_size(new_batch_size);
  mat_vector.clear();
  mat_vector.resize(new_batch_size * num_iter);
  label_vector.clear();
  label_vector.resize(new_batch_size * num_iter);
  for (int i = 0; i < new_batch_size*num_iter; ++i) {
    mat_vector[i] = cv::Mat(this->height_, this->width_, CV_8UC4);
    label_vector[i] = i;
    cv::randu(mat_vector[i], cv::Scalar::all(0), cv::Scalar::all(255));
  }
  layer.AddMatVector(mat_vector, label_vector);

  // finally consume new data and check if everything is fine
  for (int iter = 0; iter < num_iter; ++iter) {
    int offset = new_batch_size * iter;
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(new_batch_size, this->blob_top_vec_[0]->num());
    EXPECT_EQ(new_batch_size, this->blob_top_vec_[1]->num());
    const Dtype* data = this->data_blob_->cpu_data();
    for (int i = 0; i < new_batch_size; ++i) {
      EXPECT_EQ(offset + i, this->label_blob_->cpu_data()[i]);
      for (int h = 0; h < this->height_; ++h) {
        const unsigned char* ptr_mat = mat_vector[offset + i].ptr<uchar>(h);
        int index = 0;
        for (int w = 0; w < this->width_; ++w) {
          for (int c = 0; c < this->channels_; ++c) {
            data_index = (i*count) + (c * this->height_ + h) * this->width_ + w;
            Dtype pixel = static_cast<Dtype>(ptr_mat[index++]);
            EXPECT_EQ(static_cast<int>(pixel), data[data_index]);
          }
        }
      }
    }
  }
}
#endif  // USE_OPENCV
}  // namespace caffe
