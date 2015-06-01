#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/array.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

static ArrayShape shape() {
  return make_shape(3, 2, 1);
}
template <typename TypeParam>
class ArrayTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  Array<Dtype> a, b, c;

 public:
  ArrayTest():a(shape()), b(shape()), c(shape()) {
    caffe_rng_uniform<Dtype>(count(a.shape()), -1, 1, a.mutable_cpu_data());
    caffe_rng_uniform<Dtype>(count(b.shape()), -1, 1, b.mutable_cpu_data());
    caffe_rng_uniform<Dtype>(count(c.shape()), -1, 1, c.mutable_cpu_data());
  }
#define TEST_OP(n, O, OO) void test##n() {\
    c = a O b;\
    for (int i = 0; i < count(c.shape()); i++)\
      EXPECT_NEAR(c.cpu_data()[i], a.cpu_data()[i] O b.cpu_data()[i], 1e-3);\
    a OO b;\
    for (int i = 0; i < count(c.shape()); i++)\
      EXPECT_NEAR(c.cpu_data()[i], a.cpu_data()[i], 1e-3);\
  }
  TEST_OP(Add, +, +=);
  TEST_OP(Sub, -, -=);
  TEST_OP(Mul, /, /=);
  TEST_OP(Div, *, *=);
#undef TEST_OP
  void testComposite() {
    c = (Dtype)0.5*a + (Dtype)0.1*b;
    for (int i = 0; i < count(c.shape()); i++)
      EXPECT_NEAR(c.cpu_data()[i], 0.5*a.cpu_data()[i] +
                                   0.1*b.cpu_data()[i], 1e-3);
    a += (Dtype)0.1*b - (Dtype)0.5*a;
    for (int i = 0; i < count(c.shape()); i++)
      EXPECT_NEAR(c.cpu_data()[i], a.cpu_data()[i], 1e-3);
  }
};

TYPED_TEST_CASE(ArrayTest, TestDtypesAndDevices);

#define TEST_ARRAY(n) TYPED_TEST(ArrayTest, Test##n) {\
  this->test##n();\
}
TEST_ARRAY(Add);
TEST_ARRAY(Sub);
TEST_ARRAY(Mul);
TEST_ARRAY(Div);
TEST_ARRAY(Composite);
#undef TEST_OP

}  // namespace caffe
