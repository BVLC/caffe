#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/filler.hpp"

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
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
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
  layer->DataLayerSetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Reset(this->data_->mutable_cpu_data(),
      this->labels_->mutable_cpu_data(), this->data_->num());
  for (int i = 0; i < this->batches_ * 6; ++i) {
    int batch_num = i % this->batches_;
    layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
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

TYPED_TEST(MemoryDataLayerTest, AddDatumVectorDefaultTransform) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter param;
  MemoryDataParameter* memory_data_param = param.mutable_memory_data_param();
  memory_data_param->set_batch_size(this->batch_size_);
  memory_data_param->set_channels(this->channels_);
  memory_data_param->set_height(this->height_);
  memory_data_param->set_width(this->width_);
  MemoryDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);

  vector<Datum> datum_vector(this->batch_size_);
  const size_t count = this->channels_ * this->height_ * this->width_;
  size_t pixel_index = 0;
  for (int i = 0; i < this->batch_size_; ++i) {
    LOG(ERROR) << "i " << i;
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
  for (int iter = 0; iter < 5; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    const Dtype* data = this->data_blob_->cpu_data();
    size_t index = 0;
    for (int i = 0; i < this->batch_size_; ++i) {
      const string& data_string = datum_vector[i].data();
      EXPECT_EQ(i, this->label_blob_->cpu_data()[i]);
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

}  // namespace caffe
