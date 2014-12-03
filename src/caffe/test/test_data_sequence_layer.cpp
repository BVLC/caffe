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
class DataSequenceLayerTest : public MultiDeviceTest<TypeParam> {
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

    virtual std::vector<index_type> indices() {
      std::vector<index_type> result;
      result.push_back(1);
      result.push_back(4);
      result.push_back(22);
      result.push_back(3);
      result.push_back(10);
      return result;
    }
  };

 protected:
  shared_ptr<DataSource<Dtype> > data_source_;
  std::vector<Blob<Dtype>*> top_, bottom_;

 protected:
  DataSequenceLayerTest()
    : data_source_(new DummyDataSource(DataSourceParameter())) {
    top_.push_back(new Blob<Dtype>());
  }

  ~DataSequenceLayerTest() {
    for (int i = 0; i < bottom_.size(); ++i) {
      delete top_[i];
    }
  }
};

TYPED_TEST_CASE(DataSequenceLayerTest, TestDtypesAndDevices);

TYPED_TEST(DataSequenceLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter param;
  param.set_type(caffe::LayerParameter_LayerType_DATA_SEQUENCE);
  DataSequenceParameter* ds_param = param.mutable_data_sequence_param();
  ds_param->set_channels(3);
  ds_param->set_height(2);
  ds_param->set_batch_size(3);

  DataSequenceLayer<Dtype> layer(param, this->data_source_);
  layer.SetUp(this->bottom_, this->top_);

  std::vector<index_type> retrieved_indices;

  for (int round = 0; round < 3; ++round) {
    layer.Forward(this->bottom_, this->top_);

    for (int i = 0; i < this->top_.size(); ++i) {
      const Blob<Dtype>& t = *this->top_[i];
      int num = t.num();
      for (int j = 0; j < num; ++j) {
        Dtype value = t.data_at(j, 0, 0, 0);
        CHECK_EQ(value + 5, t.data_at(j, 2, 1, 0));
        retrieved_indices.push_back(static_cast<index_type>(value));
      }
    }
  }

  // Now, check that the whole range of indices is covered and in order
  std::vector<index_type> indices = this->data_source_->indices();
  std::sort(indices.begin(), indices.end());
  CHECK(std::equal(indices.begin(), indices.end(),
                   retrieved_indices.begin()));
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

TYPED_TEST(DataSequenceLayerTest, TestForwardZero) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter param;
  param.set_type(caffe::LayerParameter_LayerType_DATA_SEQUENCE);
  DataSequenceParameter* ds_param = param.mutable_data_sequence_param();
  ds_param->mutable_data_source()
      ->set_type(caffe::DataSourceParameter_DataSourceType_CONSTANT);
  ds_param->set_channels(3);
  ds_param->set_height(2);
  ds_param->set_width(9);
  ds_param->set_batch_size(3);

  DataSequenceLayer<Dtype> layer(param);

  layer.SetUp(this->bottom_, this->top_);

  for (int round = 0; round < 3; ++round) {
    layer.Forward(this->bottom_, this->top_);

    for (int i = 0; i < this->top_.size(); ++i) {
      const Blob<Dtype>& t = *this->top_[i];
      const Dtype* buffer = t.cpu_data();
      const int length = t.num() * t.channels() * t.height() * t.width();
      CHECK(all_zero(buffer, buffer + length));
    }
  }
}
}  //  namespace caffe
