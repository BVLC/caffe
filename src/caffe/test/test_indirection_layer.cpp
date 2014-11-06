#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/indexed_data.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename TypeParam>
class IndirectionLayerTest : public MultiDeviceTest<TypeParam> {
 public:
  typedef typename TypeParam::Dtype Dtype;

 protected:
  IndirectionLayerTest():
    blob_bottom_label_(new Blob<Dtype>()),
    blob_top_data_(new Blob<Dtype>())
  {}

  virtual ~IndirectionLayerTest() {
    delete blob_top_data_;
    delete blob_bottom_label_;
  }

  virtual void SetUp() {
    MakeTempDir(&filename_);
    filename_.append("/source_file");
    top_.clear();
    top_.push_back(blob_top_data_);
    bottom_.clear();
    bottom_.push_back(blob_bottom_label_);
  }

  void Fill() {
    std::ofstream out(filename_.c_str());
    blob_bottom_label_->Reshape(10, 1, 1, 1);
    Dtype* label = blob_bottom_label_->mutable_cpu_data();

    for (int i = 0; i < 10; ++i) {
      out << 0 << ' ' << 9 - i << '\n';
      label[i] = i;
    }
  }

  void TestOutput() {
    LayerParameter layerParam;
    IndirectionParameter* param = layerParam.mutable_indirection_param();
    *param->add_source() = filename_;
    param->set_channels(2);
    param->set_height(1);
    param->set_width(1);

    IndirectionLayer<Dtype> layer(layerParam);
    layer.SetUp(bottom_, top_);
    layer.Forward(bottom_, top_);

    const Dtype* data = blob_top_data_->cpu_data();
    for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(data[2 * i], 0);
      EXPECT_EQ(data[2 * i + 1] + i, 9);
    }
  }

 private:
  std::string filename_;

  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_bottom_label_;
  vector<Blob<Dtype>*> bottom_;
  vector<Blob<Dtype>*> top_;
};

TYPED_TEST_CASE(IndirectionLayerTest, TestDtypesAndDevices);

TYPED_TEST(IndirectionLayerTest, TestOutput) {
  this->Fill();
  this->TestOutput();
}
}  // namespace caffe
