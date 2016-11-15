#include <string>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/memory_sparse_data_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class MemorySparseDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MemorySparseDataLayerTest()
    : data_(new SparseBlob<Dtype>()),
      labels_(new Blob<Dtype>()),
      data_blob_(new SparseBlob<Dtype>()),
      label_blob_(new Blob<Dtype>()) {}


  void Fill(vector<SparseDatum> &datum_vector,Blob<Dtype> *labels) {
    for (int i = 0; i < 6; ++i) {
      SparseDatum datum;
      datum.set_label(i);
      datum.set_size(6);
      datum.set_nnz(i + 1);
      for (int j = 0; j < i + 1; ++j) {
        datum.mutable_data()->Add(j + 1);
        datum.mutable_indices()->Add(j);
      }
      datum_vector.push_back(datum);
    }
  }

  virtual void SetUp() {
    batch_size_ = 6;
    batches_ = 1;
    channels_ = 6;
    height_ = 1;
    width_ = 1;
    blob_top_vec_.push_back(data_blob_);
    blob_top_vec_.push_back(label_blob_);
    Fill(datum_vec_,labels_);
  }

  virtual ~MemorySparseDataLayerTest() {
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
  SparseBlob<Dtype>* const data_;
  Blob<Dtype>* const labels_;
  // blobs for the top of MemorySparseDataLayer
  SparseBlob<Dtype>* const data_blob_;
  Blob<Dtype>* const label_blob_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<SparseDatum> datum_vec_;
};

TYPED_TEST_CASE(MemorySparseDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(MemorySparseDataLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  MemoryDataParameter* md_param = layer_param.mutable_memory_data_param();
  md_param->set_batch_size(this->batch_size_);
  md_param->set_channels(this->channels_);
  md_param->set_height(this->height_);
  md_param->set_width(this->width_);
  shared_ptr<Layer<Dtype> > layer(
      new MemorySparseDataLayer<Dtype>(layer_param));
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
TYPED_TEST(MemorySparseDataLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  MemoryDataParameter* md_param = layer_param.mutable_memory_data_param();
  md_param->set_batch_size(this->batch_size_);
  md_param->set_channels(this->channels_);
  md_param->set_height(this->height_);
  md_param->set_width(this->width_);
  shared_ptr<MemorySparseDataLayer<Dtype> > layer(
      new MemorySparseDataLayer<Dtype>(layer_param));
  layer->DataLayerSetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->AddDatumVector(this->datum_vec_);
  for (int iter = 0; iter < 100; ++iter) {
      layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      for (int i = 0; i < 6; ++i) {
	//EXPECT_EQ(i, this->label_blob_->cpu_data()[i]);
	EXPECT_EQ(i, this->blob_top_vec_[1]->cpu_data()[i]);
      }
      EXPECT_EQ(0, this->data_blob_->cpu_ptr()[0]);
      for (int i = 0; i < 6; ++i) {
        EXPECT_EQ((i+1) * (i+2)/2,
		  this->data_blob_->cpu_ptr()[i+1]) << "debug ptr: iter " << iter
						    << " i " << i;
	for (int j = 0; j < i; ++j) {
	    EXPECT_EQ(j+1, this->data_blob_->
		      cpu_data()[this->data_blob_->cpu_ptr()[i]+j]) << "debug data: iter "
								    << iter << " i " << i
								    << " j " << j;
	    EXPECT_EQ(j, this->data_blob_->
		      cpu_indices()[this->data_blob_->cpu_ptr()[i]+j])
              << "debug indices: iter " << iter << " i " << i << " j " << j;
	}
      }
  }

}

}  // namespace caffe
