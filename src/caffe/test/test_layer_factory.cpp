#include <map>
#include <string>

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class LayerFactoryTest : public MultiDeviceTest<TypeParam> {};

TYPED_TEST_CASE(LayerFactoryTest, TestDtypesAndDevices);

TYPED_TEST(LayerFactoryTest, TestCreateLayer) {
  typedef typename TypeParam::Dtype Dtype;
  typename LayerRegistry<Dtype>::CreatorRegistry& registry =
      LayerRegistry<Dtype>::Registry();
  shared_ptr<Layer<Dtype> > layer;
  LayerParameter layer_param;
  for (typename LayerRegistry<Dtype>::CreatorRegistry::iterator iter =
       registry.begin(); iter != registry.end(); ++iter) {
    // Special case: PythonLayer is checked by pytest
    if (iter->first == "Python") { continue; }
    layer_param.set_type(iter->first);
    layer = LayerRegistry<Dtype>::CreateLayer(layer_param);
    EXPECT_EQ(iter->first, layer->type());
  }
}

}  // namespace caffe
