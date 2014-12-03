#include <algorithm>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/data_sources.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class DataMappingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 private:
  class DummyDataSource: public DataSource<Dtype> {
   public:
    explicit DummyDataSource(const DataSourceParameter& param)
      : DataSource<Dtype>(param) {}

    virtual length_type retrieve(index_type index, Dtype* buffer,
                                 length_type buffer_length) {
      Dtype data[] = {index, index + 1, index + 2,
                      index + 3, index + 4, index + 5};
      length_type copy_length =
          std::min<length_type>(buffer_length, sizeof(data) / sizeof(*data));
      std::copy(data, data + copy_length, buffer);
      return sizeof(data) / sizeof(*data);
    }
  };

 protected:
  shared_ptr<DataSource<Dtype> > data_source_;
  std::vector<Blob<Dtype>*> top_, bottom_;

 protected:
  DataMappingLayerTest()
    : data_source_(new DummyDataSource(DataSourceParameter())) {
    for (int i = 0; i < 4; ++i) {
      bottom_.push_back(new Blob<Dtype>(i * 4 + 1, 1, 1, 1));
      top_.push_back(new Blob<Dtype>());
    }
  }

  ~DataMappingLayerTest() {
    for (int i = 0; i < bottom_.size(); ++i) {
      delete bottom_[i];
    }

    for (int i = 0; i < bottom_.size(); ++i) {
      delete top_[i];
    }
  }

  void FillBottomRandomly(int maximum) {
    rng_t& rng = *caffe_rng();
    for (int i = 0; i < bottom_.size(); ++i) {
      Blob<Dtype>* b = bottom_[i];
      Dtype* out = b->mutable_cpu_data();
      for (int j = 0; j < b->num(); ++j) {
        out[j] = rng() % maximum;
      }
    }
  }

  void CheckTopAgainstBottom(const DataMappingParameter& param) {
    CHECK_EQ(bottom_.size(), top_.size());
    for (int i = 0; i < bottom_.size(); ++i) {
      const Blob<Dtype>& b = *bottom_[i];
      const Blob<Dtype>& t = *top_[i];

      CHECK_EQ(t.num(), b.num());
      CHECK_EQ(t.channels(), param.channels());
      CHECK_EQ(t.height(), param.height());
      CHECK_EQ(t.width(), param.width());
    }
  }
};

TYPED_TEST_CASE(DataMappingLayerTest, TestDtypesAndDevices);

TYPED_TEST(DataMappingLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;

  this->FillBottomRandomly(65535);

  LayerParameter param;
  param.set_type(caffe::LayerParameter_LayerType_DATA_MAPPING);
  DataMappingParameter* dm_param = param.mutable_data_mapping_param();
  dm_param->set_channels(3);
  dm_param->set_height(2);

  DataMappingLayer<Dtype> layer(param, this->data_source_);

  layer.SetUp(this->bottom_, this->top_);
  this->CheckTopAgainstBottom(param.data_mapping_param());
  layer.Forward(this->bottom_, this->top_);

  for (int i = 0; i < this->bottom_.size(); ++i) {
    const Blob<Dtype>& b = *this->bottom_[i];
    const Blob<Dtype>& t = *this->top_[i];

    int num = b.num();

    for (int j = 0; j < num; ++j) {
      CHECK_EQ(b.data_at(j, 0, 0, 0), t.data_at(j, 0, 0, 0));
      CHECK_EQ(b.data_at(j, 0, 0, 0) + 5, t.data_at(j, 2, 1, 0));
    }
  }
}

template <typename Iterator>
static bool all_zero(Iterator begin, Iterator end) {
  while (begin != end) {
    if (*begin != 0)
      return false;
    ++begin;
  }
  return true;
}

TYPED_TEST(DataMappingLayerTest, TestForwardZero) {
  typedef typename TypeParam::Dtype Dtype;

  this->FillBottomRandomly(65535);

  LayerParameter param;
  param.set_type(caffe::LayerParameter_LayerType_DATA_MAPPING);
  DataMappingParameter* dm_param = param.mutable_data_mapping_param();
  dm_param->mutable_data_source()
      ->set_type(caffe::DataSourceParameter_DataSourceType_CONSTANT);
  dm_param->set_channels(3);
  dm_param->set_height(2);
  dm_param->set_width(9);

  DataMappingLayer<Dtype> layer(param);

  layer.SetUp(this->bottom_, this->top_);
  this->CheckTopAgainstBottom(param.data_mapping_param());
  layer.Forward(this->bottom_, this->top_);

  for (int i = 0; i < this->bottom_.size(); ++i) {
    const Blob<Dtype>& t = *this->top_[i];
    const Dtype* buffer = t.cpu_data();
    const int length = t.num() * t.channels() * t.height() * t.width();
    CHECK(all_zero(buffer, buffer + length));
  }
}
}  //  namespace caffe
