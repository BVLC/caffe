#include <map>
#include <string>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class ImportsTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitNetFromProtoString(const string& proto) {
    NetParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    net_.reset(new Net<Dtype>(param));
  }

  virtual void InitNet() {
    string file = CMAKE_SOURCE_DIR "caffe/test/test_data/module.prototxt";
    string proto =
        "name: 'TestNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    num: 5 "
        "    channels: 1 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "  top: 'label' "
        "} "
        "layers: { "
        "  name: 'import' "
        "  type: IMPORT "
        "  import_param { "
        "    net: '" + file + "' "
        "    var { name: 'num_output' value: '1000' } "
        "  } "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: SOFTMAX_LOSS "
        "  bottom: 'import/innerproduct' "
        "  bottom: 'label' "
        "  top: 'top_loss' "
        "} ";
    InitNetFromProtoString(proto);
  }

  shared_ptr<Net<Dtype> > net_;
};

TYPED_TEST_CASE(ImportsTest, TestDtypesAndDevices);

TYPED_TEST(ImportsTest, ConvPool) {
  this->InitNet();
  EXPECT_TRUE(this->net_->has_blob("data"));
  EXPECT_TRUE(this->net_->has_blob("label"));
  EXPECT_TRUE(this->net_->has_blob("import/innerproduct"));
  EXPECT_FALSE(this->net_->has_blob("loss"));
}
}  // namespace caffe
