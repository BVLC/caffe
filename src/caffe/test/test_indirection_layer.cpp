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
    Fill();
  }

  void Fill() {
    blob_bottom_label_->Reshape(10, 1, 1, 1);
    Dtype* label = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < 10; ++i)
      label[i] = i;
  }

  void FillWrong() {
    blob_bottom_label_->Reshape(10, 1, 1, 1);
    Dtype* label = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < 10; ++i)
      label[i] = i + 1;
  }

  void FillText() {
    std::ofstream out(filename_.c_str());
    CHECK(out);

    for (int i = 0; i < 10; ++i) {
      out << 0 << ' ' << 9 - i << '\n';
    }
  }

  void FillBinary() {
    std::ofstream out(filename_.c_str());
    CHECK(out);

    for (int i = 0; i < 10; ++i) {
      std::ostringstream os;
      os << filename_ << '_' << i << ".bin";
      const std::string& bin_fn = os.str();

      std::ofstream bin_out(bin_fn.c_str(), std::ios_base::binary);
      CHECK(bin_out);

      Dtype d = 0;
      bin_out.write(reinterpret_cast<char*>(&d), sizeof(d));
      d = 9 - i;
      bin_out.write(reinterpret_cast<char*>(&d), sizeof(d));
      CHECK_EQ(bin_out.tellp(), 2 * sizeof(Dtype));
      out << bin_fn << '\n';
    }
  }

  void FillBlobs() {
    std::ofstream out(filename_.c_str());
    CHECK(out);

    BlobProto blob;
    blob.set_channels(2);
    blob.set_height(1);
    blob.set_width(1);
    blob.add_data(0);
    blob.add_data(0);
    float* data = blob.mutable_data()->mutable_data();

    for (int i = 0; i < 10; ++i) {
      data[1] = 9 - i;
      std::ostringstream os;
      os << filename_ << '_' << i << ".binaryproto";
      const std::string& proto_fn = os.str();
      WriteProtoToBinaryFile(blob, proto_fn);
      out << proto_fn << '\n';
    }
  }

  void TestOutput(IndirectionParameter_IndirectionSourceType type,
                  IndirectionParameter_IndirectionCacheType cache_type) {
    LayerParameter layerParam;
    IndirectionParameter* param = layerParam.mutable_indirection_param();
    param->set_type(type);
    param->set_cache_type(cache_type);
    *param->add_source() = filename_;
    param->set_channels(2);
    param->set_height(1);
    param->set_width(1);
    param->set_cache_block_size(3);
    param->set_cache_block_num(2);

    IndirectionLayer<Dtype> layer(layerParam);
    layer.SetUp(bottom_, top_);
    layer.Forward(bottom_, top_);

    const Dtype* data = blob_top_data_->cpu_data();
    for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(data[2 * i], 0);
      EXPECT_EQ(data[2 * i + 1] + i, 9);
    }

    // Test it twice to see if the caching is right
    layer.Forward(bottom_, top_);

    data = blob_top_data_->cpu_data();
    for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(data[2 * i], 0);
      EXPECT_EQ(data[2 * i + 1] + i, 9);
    }
  }

  void TestOutput(IndirectionParameter_IndirectionSourceType type) {
    TestOutput(type, IndirectionParameter_IndirectionCacheType_NONE);
    TestOutput(type, IndirectionParameter_IndirectionCacheType_WHOLE);
    TestOutput(type, IndirectionParameter_IndirectionCacheType_CLOCK);
  }

 private:
  std::string filename_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> bottom_;
  vector<Blob<Dtype>*> top_;
};

TYPED_TEST_CASE(IndirectionLayerTest, TestDtypesAndDevices);

TYPED_TEST(IndirectionLayerTest, TestTextFile) {
  this->FillText();
  this->TestOutput(IndirectionParameter_IndirectionSourceType_SIMPLE_TEXT);
}

TYPED_TEST(IndirectionLayerTest, TestBinaryFiles) {
  this->FillBinary();
  this->TestOutput(IndirectionParameter_IndirectionSourceType_INDEXED_BINARY);
}

TYPED_TEST(IndirectionLayerTest, TestBlobProtos) {
  this->FillBlobs();
  this->TestOutput(IndirectionParameter_IndirectionSourceType_INDEXED_BLOB);
}
}  // namespace caffe
