#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/tensor.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template<typename Dtype>
class TensorTest : public ::testing::Test {
 protected:
  TensorTest()
      : tensor_(),
        tensor_preshaped_(2, 3, 4, 5) {
  }
  virtual ~TensorTest() {
  }
  SyncedTensor<Dtype> tensor_;
  SyncedTensor<Dtype> tensor_preshaped_;
};

TYPED_TEST_CASE(TensorTest, TestDtypes);


TYPED_TEST(TensorTest, TestInitialization) {
//  EXPECT_TRUE(this->tensor_);EXPECT_TRUE(this->tensor_preshaped_);
  EXPECT_EQ(this->tensor_preshaped_.num_axes(), 4);
  EXPECT_EQ(
    this->tensor_preshaped_.shape(0), 2);
  EXPECT_EQ(
    this->tensor_preshaped_.shape(1), 3);
  EXPECT_EQ(
    this->tensor_preshaped_.shape(2), 4);
  EXPECT_EQ(
    this->tensor_preshaped_.shape(3), 5);
  EXPECT_EQ(
    this->tensor_preshaped_.size(), 120);
  EXPECT_EQ(this->tensor_.num_axes(), 0);
  EXPECT_EQ(
    this->tensor_.size(), 0);
}

TYPED_TEST(TensorTest, TestPointersCPUGPU) {
  EXPECT_TRUE(
    this->tensor_preshaped_.cpu_data());
  EXPECT_TRUE(
    this->tensor_preshaped_.mutable_cpu_data());
#ifndef CPU_ONLY
  EXPECT_TRUE(this->tensor_preshaped_.gpu_data());
  EXPECT_TRUE(
    this->tensor_preshaped_.mutable_gpu_data());
#else
  EXPECT_FALSE(this->tensor_preshaped_.gpu_data());
  EXPECT_FALSE(
    this->tensor_preshaped_.mutable_gpu_data());
#endif
}

TYPED_TEST(TensorTest, TestReshape) {
  this->tensor_.Reshape(2, 3, 4, 5);
  EXPECT_EQ(this->tensor_.shape(0), 2);
  EXPECT_EQ(this->tensor_.shape(1), 3);
  EXPECT_EQ(
    this->tensor_.shape(2), 4);
  EXPECT_EQ(this->tensor_.shape(3), 5);
  EXPECT_EQ(
    this->tensor_.size(), 120);}

/*
 TYPED_TEST(TensorTest, TestLegacyTensorProtoShapeEquals) {
 TensorProto tensor_proto;

 // Reshape to (3 x 2).
 vector<int> shape(2);
 shape[0] = 3;
 shape[1] = 2;
 this->tensor_.Reshape(shape);

 tensor_proto.mutable_shape()->add_dim(3);
 tensor_proto.mutable_shape()->add_dim(2);
 EXPECT_TRUE(this->tensor_.ShapeEquals(tensor_proto));

 // (3 x 2) tensor != (3 x 2 x 1) tensor
 tensor_proto.clear_shape();
 tensor_proto.mutable_shape()->add_dim(3);
 tensor_proto.mutable_shape()->add_dim(2);
 tensor_proto.mutable_shape()->add_dim(1);
 EXPECT_FALSE(this->tensor_.ShapeEquals(tensor_proto));

 // (3 x 2) tensor != (3 x 1 x 2) tensor
 tensor_proto.clear_shape();
 tensor_proto.mutable_shape()->add_dim(3);
 tensor_proto.mutable_shape()->add_dim(2);
 tensor_proto.mutable_shape()->add_dim(1);
 EXPECT_FALSE(this->tensor_.ShapeEquals(tensor_proto));

 // (3 x 2) tensor != (1 x 3 x 2) tensor
 tensor_proto.clear_shape();
 tensor_proto.mutable_shape()->add_dim(1);
 tensor_proto.mutable_shape()->add_dim(3);
 tensor_proto.mutable_shape()->add_dim(2);
 EXPECT_FALSE(this->tensor_.ShapeEquals(tensor_proto));

 // Reshape to (1 x 3 x 2).
 shape.insert(shape.begin(), 1);
 this->tensor_.Reshape(shape);

 // (1 x 3 x 2) tensor != (1 x 3 x 2) tensor
 tensor_proto.clear_shape();
 tensor_proto.mutable_shape()->add_dim(1);
 tensor_proto.mutable_shape()->add_dim(1);
 tensor_proto.mutable_shape()->add_dim(3);
 tensor_proto.mutable_shape()->add_dim(2);
 EXPECT_FALSE(this->tensor_.ShapeEquals(tensor_proto));

 // Reshape to (2 x 3 x 2).
 shape[0] = 2;
 this->tensor_.Reshape(shape);

 // (2 x 3 x 2) tensor != (1 x 1 x 3 x 2) tensor
 tensor_proto.clear_shape();
 tensor_proto.mutable_shape()->add_dim(1);
 tensor_proto.mutable_shape()->add_dim(1);
 tensor_proto.mutable_shape()->add_dim(3);
 tensor_proto.mutable_shape()->add_dim(2);
 EXPECT_FALSE(this->tensor_.ShapeEquals(tensor_proto));}
 }
 */

template<typename TypeParam>
class TensorMathTest : public MultiDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;
 protected:
TensorMathTest()
    : tensor_(2, 3, 4, 5),
      epsilon_(1e-6) {
}

virtual ~TensorMathTest() {
}
SyncedTensor<Dtype> tensor_;
Dtype epsilon_;
};

TYPED_TEST_CASE(TensorMathTest, TestDtypesAndDevices);

/*
TYPED_TEST(TensorMathTest, TestOnes) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  SyncedTensor<Dtype> result(this->tensor_.shape());
  filler.Fill(&result);
  Caffe::set_mode(TypeParam::device);Tensor<Dtype>::ones(&result, this->tensor_.shape());
  Dtype expected_result = 1;
  const Dtype* data = result.cpu_data();
  for (int i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(
  expected_result, data[i], this->epsilon_);
}
}
*/

TYPED_TEST(TensorMathTest, TestZeros) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  SyncedTensor<Dtype> result(this->tensor_.shape());
  filler.Fill(&result);
  Caffe::set_mode(TypeParam::device);
  result.zeros();
  Dtype expected_result = 0;
  const Dtype* data = result.cpu_data();
  for (int i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(expected_result, data[i], this->epsilon_);
}
}


TYPED_TEST(TensorMathTest, TestAbs) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&(this->tensor_));
  SyncedTensor<Dtype> result;
  Caffe::set_mode(TypeParam::device);
  result.abs(this->tensor_);
  const Dtype* data = this->tensor_.cpu_data();
  const Dtype* result_data = result.cpu_data();
  for (size_t i = 0; i < this->tensor_.size(); ++i) {
    EXPECT_NEAR(std::abs(data[i]), result_data[i], this->epsilon_);
}
}

TYPED_TEST(TensorMathTest, TestPow) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
  FillerParameter filler_param;
  filler_param.set_min(0);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&(this->tensor_));
  SyncedTensor<Dtype> result;
  Caffe::set_mode(TypeParam::device);
  const Dtype value = 3.14;
  result.pow(this->tensor_, value);
  const Dtype* data = this->tensor_.cpu_data();
  const Dtype* result_data = result.cpu_data();
  Dtype expected_result;
  for (size_t i = 0; i < this->tensor_.size(); ++i) {
    expected_result = std::pow(data[i], value);
    EXPECT_NEAR(expected_result, result_data[i],
        this->epsilon_ * expected_result);
  }
}

TYPED_TEST(TensorMathTest, TestAddValue) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  SyncedTensor<Dtype> result(this->tensor_.shape());
  filler.Fill(&result);
  const Dtype* result_data = result.cpu_data();
  vector<Dtype> old_result_data(result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    old_result_data[i] = result_data[i];
  }
  const Dtype value = 3.14;
  Caffe::set_mode(TypeParam::device);
  result.add(value);
  for (size_t i = 0; i < this->tensor_.size(); ++i) {
    EXPECT_NEAR(old_result_data[i] + value, result_data[i],
                                               this->epsilon_);
}
}

TYPED_TEST(TensorMathTest, TestAddTensor) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&(this->tensor_));
  SyncedTensor<Dtype> result(this->tensor_.shape());
  filler.Fill(&result);
  const Dtype* result_data = result.cpu_data();
  vector<Dtype> old_result_data(result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    old_result_data[i] = result_data[i];
  }
  Caffe::set_mode(TypeParam::device);
  result.add(this->tensor_);
  const Dtype* data = this->tensor_.cpu_data();
  for (size_t i = 0; i < this->tensor_.size(); ++i) {
    EXPECT_NEAR(data[i] + old_result_data[i], result_data[i], this->epsilon_);
}
}

TYPED_TEST(TensorMathTest, TestAddValueMultipliedTensor) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&(this->tensor_));
  const Dtype value = 3.14;
  SyncedTensor<Dtype> result(this->tensor_.shape());
  filler.Fill(&result);
  const Dtype* result_data = result.cpu_data();
  vector<Dtype> old_result_data(result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    old_result_data[i] = result_data[i];
  }
  Caffe::set_mode(TypeParam::device);
  result.add(value, this->tensor_);
  const Dtype* data = this->tensor_.cpu_data();
  for (size_t i = 0; i < this->tensor_.size(); ++i) {
    EXPECT_NEAR(
  old_result_data[i] + value * data[i], result_data[i], this->epsilon_);
}
}

TYPED_TEST(TensorMathTest, TestDot) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&(this->tensor_));
  SyncedTensor<Dtype> result(this->tensor_.shape());
  filler.Fill(&result);
  const Dtype* data = this->tensor_.cpu_data();
  const Dtype* result_data = result.cpu_data();
  Dtype expected_result = 0;
  for (size_t i = 0; i < this->tensor_.size(); ++i) {
    expected_result += data[i] * result_data[i];
  }
  Dtype epsilon = std::abs(this->epsilon_ * expected_result);
  if (sizeof(Dtype) == sizeof(float)) {
    epsilon *= 3;
  }
  Caffe::set_mode(TypeParam::device);
  EXPECT_NEAR(expected_result, result.dot(this->tensor_), epsilon);
}

TYPED_TEST(TensorMathTest, TestMul) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  SyncedTensor<Dtype> result(this->tensor_.shape());
  filler.Fill(&result);
  const Dtype* result_data = result.cpu_data();
  vector<Dtype> old_result_data(result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    old_result_data[i] = result_data[i];
  }
  const Dtype value = 3.14;
  Caffe::set_mode(TypeParam::device);
  result.mul(value);
  for (size_t i = 0; i < this->tensor_.size(); ++i) {
    EXPECT_NEAR(
  old_result_data[i] * value, result_data[i], this->epsilon_);
}
}
TYPED_TEST(TensorMathTest, TestDiv) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  SyncedTensor<Dtype> result(this->tensor_.shape());
  filler.Fill(&result);
  vector<Dtype> old_result_data(result.size());
  const Dtype* result_data = result.cpu_data();
  for (size_t i = 0; i < result.size(); ++i) {
    old_result_data[i] = result_data[i];
  }
  const Dtype value = 3.14;
  Caffe::set_mode(TypeParam::device);
  result.div(value);
  for (size_t i = 0; i < this->tensor_.size(); ++i) {
    EXPECT_NEAR(
  old_result_data[i] / value, result_data[i], this->epsilon_);
}
}

TYPED_TEST(TensorMathTest, TestCmul) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&(this->tensor_));
  SyncedTensor<Dtype> that(this->tensor_.shape());
  filler.Fill(&that);
  SyncedTensor<Dtype> result(this->tensor_.shape());
  Caffe::set_mode(TypeParam::device);
result.cmul(this->tensor_, that);
  const Dtype* data = this->tensor_.cpu_data();
  const Dtype* that_data = that.cpu_data();
  const Dtype* result_data = result.cpu_data();
  for (size_t i = 0; i < this->tensor_.size(); ++i) {
    EXPECT_NEAR(
        data[i] * that_data[i], result_data[i], this->epsilon_);
}
}

TYPED_TEST(TensorMathTest, TestCdiv) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&(this->tensor_));
  SyncedTensor<Dtype> that(this->tensor_.shape());
  filler.Fill(&that);
  SyncedTensor<Dtype> result(this->tensor_.shape());
  Caffe::set_mode(TypeParam::device);
result.cdiv(this->tensor_, that);
  const Dtype* data = this->tensor_.cpu_data();
  const Dtype* that_data = that.cpu_data();
  const Dtype* result_data = result.cpu_data();
  for (size_t i = 0; i < this->tensor_.size(); ++i) {
    EXPECT_NEAR(
        data[i] / that_data[i], result_data[i], this->epsilon_);
}
}

TYPED_TEST(TensorMathTest, TestMv) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&(this->tensor_));
  SyncedTensor<Dtype> result(this->tensor_.shape());
  filler.Fill(&result);
  Caffe::set_mode(TypeParam::device);
//  result.mv(this->tensor_);
  const Dtype* data = this->tensor_.cpu_data();
  const Dtype* result_data = result.cpu_data();
  for (size_t i = 0; i < this->tensor_.size(); ++i) {
    // TODO
    EXPECT_NEAR(
  data[i], data[i], this->epsilon_);
    EXPECT_NEAR(
        result_data[i], result_data[i], this->epsilon_);
  }
}

TYPED_TEST(TensorMathTest, TestAddmv) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&(this->tensor_));
  SyncedTensor<Dtype> result(this->tensor_.shape());
  filler.Fill(&result);
  Caffe::set_mode(TypeParam::device);
//  result.addmv(this->tensor_);
  const Dtype* data = this->tensor_.cpu_data();
  const Dtype* result_data = result.cpu_data();
  for (size_t i = 0; i < this->tensor_.size(); ++i) {
    // TODO
    EXPECT_NEAR(
  data[i], data[i], this->epsilon_);
    EXPECT_NEAR(
        result_data[i], result_data[i], this->epsilon_);
  }
}

TYPED_TEST(TensorMathTest, TestMm) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&(this->tensor_));
  SyncedTensor<Dtype> result(this->tensor_.shape());
  filler.Fill(&result);
  Caffe::set_mode(TypeParam::device);
//  result.mm(this->tensor_);
  const Dtype* data = this->tensor_.cpu_data();
  const Dtype* result_data = result.cpu_data();
  for (size_t i = 0; i < this->tensor_.size(); ++i) {
    // TODO
    EXPECT_NEAR(
  data[i], data[i], this->epsilon_);
    EXPECT_NEAR(
        result_data[i], result_data[i], this->epsilon_);
  }
}

TYPED_TEST(TensorMathTest, TestAddmm) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&(this->tensor_));
  SyncedTensor<Dtype> result(this->tensor_.shape());
  filler.Fill(&result);
  Caffe::set_mode(TypeParam::device);
//  result.addmm(this->tensor_);
  const Dtype* data = this->tensor_.cpu_data();
  const Dtype* result_data = result.cpu_data();
  for (size_t i = 0; i < this->tensor_.size(); ++i) {
    // TODO
    EXPECT_NEAR(
  data[i], data[i], this->epsilon_);
    EXPECT_NEAR(
        result_data[i], result_data[i], this->epsilon_);
  }
}

TYPED_TEST(TensorMathTest, TestAsum) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_.asum());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&(this->tensor_));
  Dtype expected_asum = 0;
  const Dtype* data = this->tensor_.cpu_data();
  for (size_t i = 0; i < this->tensor_.size(); ++i) {
    expected_asum += std::fabs(data[i]);
  }
  Caffe::set_mode(TypeParam::device);
  EXPECT_NEAR(expected_asum, this->tensor_.asum(),
      this->epsilon_ * expected_asum);
}

TYPED_TEST(TensorMathTest, TestScaleData) {
  typedef typename TypeParam::Dtype Dtype;

  EXPECT_EQ(0, this->tensor_.asum());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&(this->tensor_));
  const Dtype asum_before_scale = this->tensor_.asum();
  Caffe::set_mode(TypeParam::device);
  const Dtype kDataScaleFactor = 3;
  this->tensor_.scale(kDataScaleFactor);
  EXPECT_NEAR(
  asum_before_scale * kDataScaleFactor, this->tensor_.asum(),
  this->epsilon_ * asum_before_scale * kDataScaleFactor);
}

TYPED_TEST(TensorMathTest, TestSumsq) {
typedef typename TypeParam::Dtype Dtype;

// Uninitialized Tensor should have sum of squares == 0.
EXPECT_EQ(0, this->tensor_.sumsq());
FillerParameter filler_param;
filler_param.set_min(-3);
filler_param.set_max(3);
UniformFiller<Dtype> filler(filler_param);
filler.Fill(&(this->tensor_));
Dtype expected_sumsq = 0;
const Dtype* data = this->tensor_.cpu_data();
for (size_t i = 0; i < this->tensor_.size(); ++i) {
  expected_sumsq += data[i] * data[i];
}
Caffe::set_mode(TypeParam::device);
EXPECT_NEAR(
    expected_sumsq, this->tensor_.sumsq(), this->epsilon_ * expected_sumsq);
}

}  // namespace caffe
