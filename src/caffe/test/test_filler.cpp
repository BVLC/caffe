#include <vector>

#include "gtest/gtest.h"

#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class ConstantFillerTest : public ::testing::Test {
 protected:
  ConstantFillerTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_param_.set_value(10.);
    filler_.reset(new ConstantFiller<Dtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~ConstantFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<ConstantFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(ConstantFillerTest, TestDtypes);

TYPED_TEST(ConstantFillerTest, TestFill) {
  EXPECT_TRUE(this->blob_);
  const int count = this->blob_->count();
  const TypeParam* data = this->blob_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], this->filler_param_.value());
  }
}

template <typename Dtype>
class UniformFillerTest : public ::testing::Test {
 protected:
  UniformFillerTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_param_.set_min(1.);
    filler_param_.set_max(2.);
    filler_.reset(new UniformFiller<Dtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~UniformFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<UniformFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(UniformFillerTest, TestDtypes);

TYPED_TEST(UniformFillerTest, TestFill) {
  EXPECT_TRUE(this->blob_);
  const int count = this->blob_->count();
  const TypeParam* data = this->blob_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], this->filler_param_.min());
    EXPECT_LE(data[i], this->filler_param_.max());
  }
}

template <typename Dtype>
class UniformStaticFillerTest : public ::testing::Test {
 protected:
  UniformStaticFillerTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_param_.set_min(1.);
    filler_param_.set_max(2.);
    filler_.reset(new UniformStaticFiller<Dtype>(filler_param_));
    filler_->Fill(blob_.get());
  }
  shared_ptr<Blob<Dtype> > const blob_;
  FillerParameter filler_param_;
  shared_ptr<UniformStaticFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(UniformStaticFillerTest, TestDtypes);

TYPED_TEST(UniformStaticFillerTest, TestFill) {
  EXPECT_TRUE(this->blob_);
  const int count = this->blob_->count();
  const TypeParam* data = this->blob_->cpu_data();
  // We want to check that repeated calls to the static filler returns the same
  // values. So we copy the first filler call to data_0 and the second one to
  // data_1 and then check whether they are equal.
  std::vector<TypeParam> data_0, data_1;
  data_0.resize(count);
  data_1.resize(count);
  caffe_copy(count, data, &data_0.front());

  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], this->filler_param_.min());
    EXPECT_LE(data[i], this->filler_param_.max());
  }

  this->filler_->Fill(this->blob_.get());
  caffe_copy(count, data, &data_1.front());
  for (int i = 0; i < count; ++i) {
    // We do not use EXPECT_FLOAT_EQ because the data must match
    // bit by bit
    EXPECT_EQ(data_0[i], data_1[i]);
  }
}

template <typename Dtype>
class PositiveUnitballFillerTest : public ::testing::Test {
 protected:
  PositiveUnitballFillerTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_.reset(new PositiveUnitballFiller<Dtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~PositiveUnitballFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<PositiveUnitballFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(PositiveUnitballFillerTest, TestDtypes);

TYPED_TEST(PositiveUnitballFillerTest, TestFill) {
  EXPECT_TRUE(this->blob_);
  const int num = this->blob_->num();
  const int count = this->blob_->count();
  const int dim = count / num;
  const TypeParam* data = this->blob_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], 0);
    EXPECT_LE(data[i], 1);
  }
  for (int i = 0; i < num; ++i) {
    TypeParam sum = 0;
    for (int j = 0; j < dim; ++j) {
      sum += data[i * dim + j];
    }
    EXPECT_GE(sum, 0.999);
    EXPECT_LE(sum, 1.001);
  }
}

template <typename Dtype>
class PositiveUnitballStaticFillerTest : public ::testing::Test {
 protected:
  PositiveUnitballStaticFillerTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_.reset(new PositiveUnitballStaticFiller<Dtype>(filler_param_));
    filler_->Fill(blob_.get());
  }
  shared_ptr<Blob<Dtype> > const blob_;
  FillerParameter filler_param_;
  shared_ptr<PositiveUnitballStaticFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(PositiveUnitballStaticFillerTest, TestDtypes);

TYPED_TEST(PositiveUnitballStaticFillerTest, TestFill) {
  EXPECT_TRUE(this->blob_);
  const int num = this->blob_->num();
  const int count = this->blob_->count();
  const int dim = count / num;
  const TypeParam* data = this->blob_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], 0);
    EXPECT_LE(data[i], 1);
  }
  for (int i = 0; i < num; ++i) {
    TypeParam sum = 0;
    for (int j = 0; j < dim; ++j) {
      sum += data[i * dim + j];
    }
    EXPECT_GE(sum, 0.999);
    EXPECT_LE(sum, 1.001);
  }
  // We want to check that repeated calls to the static filler returns the same
  // values. So we copy the first filler call to data_0 and the second one to
  // data_1 and then check whether they are equal.
  std::vector<TypeParam> data_0, data_1;
  data_0.resize(count);
  data_1.resize(count);
  caffe_copy(count, data, &data_0.front());

  this->filler_->Fill(this->blob_.get());
  caffe_copy(count, data, &data_1.front());
  for (int i = 0; i < count; ++i) {
    // We do not use EXPECT_FLOAT_EQ because the data must match
    // bit by bit
    EXPECT_EQ(data_0[i], data_1[i]);
  }
}

template <typename Dtype>
class GaussianFillerTest : public ::testing::Test {
 protected:
  GaussianFillerTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_param_.set_mean(10.);
    filler_param_.set_std(0.1);
    filler_.reset(new GaussianFiller<Dtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~GaussianFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<GaussianFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(GaussianFillerTest, TestDtypes);

TYPED_TEST(GaussianFillerTest, TestFill) {
  EXPECT_TRUE(this->blob_);
  const int count = this->blob_->count();
  const TypeParam* data = this->blob_->cpu_data();
  TypeParam mean = 0.;
  TypeParam var = 0.;
  for (int i = 0; i < count; ++i) {
    mean += data[i];
    var += (data[i] - this->filler_param_.mean()) *
        (data[i] - this->filler_param_.mean());
  }
  mean /= count;
  var /= count;
  // Very loose test.
  EXPECT_GE(mean, this->filler_param_.mean() - this->filler_param_.std() * 5);
  EXPECT_LE(mean, this->filler_param_.mean() + this->filler_param_.std() * 5);
  TypeParam target_var = this->filler_param_.std() * this->filler_param_.std();
  EXPECT_GE(var, target_var / 5.);
  EXPECT_LE(var, target_var * 5.);
}


template <typename Dtype>
class GaussianStaticFillerTest : public ::testing::Test {
 protected:
  GaussianStaticFillerTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_param_.set_mean(10.);
    filler_param_.set_std(0.1);
    filler_.reset(new GaussianStaticFiller<Dtype>(filler_param_));
    filler_->Fill(blob_.get());
  }
  shared_ptr<Blob<Dtype> > const blob_;
  FillerParameter filler_param_;
  shared_ptr<GaussianStaticFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(GaussianStaticFillerTest, TestDtypes);

TYPED_TEST(GaussianStaticFillerTest, TestFill) {
  EXPECT_TRUE(this->blob_);
  const int count = this->blob_->count();
  const TypeParam* data = this->blob_->cpu_data();
  TypeParam mean = 0.;
  TypeParam var = 0.;
  for (int i = 0; i < count; ++i) {
    mean += data[i];
    var += (data[i] - this->filler_param_.mean()) *
        (data[i] - this->filler_param_.mean());
  }
  mean /= count;
  var /= count;
  // Very loose test.
  EXPECT_GE(mean, this->filler_param_.mean() - this->filler_param_.std() * 5);
  EXPECT_LE(mean, this->filler_param_.mean() + this->filler_param_.std() * 5);
  TypeParam target_var = this->filler_param_.std() * this->filler_param_.std();
  EXPECT_GE(var, target_var / 5.);
  EXPECT_LE(var, target_var * 5.);

  // We want to check that repeated calls to the static filler returns the same
  // values. So we copy the first filler call to data_0 and the second one to
  // data_1 and then check whether they are equal.
  std::vector<TypeParam> data_0, data_1;
  data_0.resize(count);
  data_1.resize(count);
  caffe_copy(count, data, &data_0.front());

  this->filler_->Fill(this->blob_.get());
  caffe_copy(count, data, &data_1.front());
  for (int i = 0; i < count; ++i) {
    // We do not use EXPECT_FLOAT_EQ because the data must match
    // bit by bit
    EXPECT_EQ(data_0[i], data_1[i]);
  }
}

template <typename Dtype>
class XavierFillerTest : public ::testing::Test {
 protected:
  XavierFillerTest()
      : blob_(new Blob<Dtype>(1000, 2, 4, 5)),
        filler_param_() {
  }
  virtual void test_params(FillerParameter_VarianceNorm variance_norm,
      Dtype n) {
    this->filler_param_.set_variance_norm(variance_norm);
    this->filler_.reset(new XavierFiller<Dtype>(this->filler_param_));
    this->filler_->Fill(blob_);
    EXPECT_TRUE(this->blob_);
    const int count = this->blob_->count();
    const Dtype* data = this->blob_->cpu_data();
    Dtype mean = 0.;
    Dtype ex2 = 0.;
    for (int i = 0; i < count; ++i) {
      mean += data[i];
      ex2 += data[i] * data[i];
    }
    mean /= count;
    ex2 /= count;
    Dtype std = sqrt(ex2 - mean*mean);
    Dtype target_std = sqrt(2.0 / n);
    EXPECT_NEAR(mean, 0.0, 0.1);
    EXPECT_NEAR(std, target_std, 0.1);
  }
  virtual ~XavierFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<XavierFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(XavierFillerTest, TestDtypes);

TYPED_TEST(XavierFillerTest, TestFillFanIn) {
  TypeParam n = 2*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_IN, n);
}
TYPED_TEST(XavierFillerTest, TestFillFanOut) {
  TypeParam n = 1000*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_OUT, n);
}
TYPED_TEST(XavierFillerTest, TestFillAverage) {
  TypeParam n = (2*4*5 + 1000*4*5) / 2.0;
  this->test_params(FillerParameter_VarianceNorm_AVERAGE, n);
}

template <typename Dtype>
class XavierStaticFillerTest : public ::testing::Test {
 protected:
  XavierStaticFillerTest()
      : blob_(new Blob<Dtype>(1000, 2, 4, 5)),
        filler_param_() {
  }
  virtual void test_params(FillerParameter_VarianceNorm variance_norm,
      Dtype n) {
    this->filler_param_.set_variance_norm(variance_norm);
    this->filler_.reset(new XavierStaticFiller<Dtype>(this->filler_param_));
    this->filler_->Fill(blob_.get());
    EXPECT_TRUE(this->blob_);
    const int count = this->blob_->count();
    const Dtype* data = this->blob_->cpu_data();
    Dtype mean = 0.;
    Dtype ex2 = 0.;
    for (int i = 0; i < count; ++i) {
      mean += data[i];
      ex2 += data[i] * data[i];
    }
    mean /= count;
    ex2 /= count;
    Dtype std = sqrt(ex2 - mean*mean);
    Dtype target_std = sqrt(2.0 / n);
    EXPECT_NEAR(mean, 0.0, 0.1);
    EXPECT_NEAR(std, target_std, 0.1);

    // We want to check that repeated calls to the static
    // filler returns the same values. So we copy the first
    // filler call to data_0 and the second one to
    // data_1 and then check whether they are equal.
    std::vector<Dtype> data_0, data_1;
    data_0.resize(count);
    data_1.resize(count);
    caffe_copy(count, data, &data_0.front());

    this->filler_->Fill(this->blob_.get());
    caffe_copy(count, data, &data_1.front());
    for (int i = 0; i < count; ++i) {
      // We do not use EXPECT_FLOAT_EQ because the data must match
      // bit by bit
      EXPECT_EQ(data_0[i], data_1[i]);
    }
  }
  shared_ptr<Blob<Dtype> > const blob_;
  FillerParameter filler_param_;
  shared_ptr<XavierStaticFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(XavierStaticFillerTest, TestDtypes);

TYPED_TEST(XavierStaticFillerTest, TestFillFanIn) {
  TypeParam n = 2*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_IN, n);
}
TYPED_TEST(XavierStaticFillerTest, TestFillFanOut) {
  TypeParam n = 1000*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_OUT, n);
}
TYPED_TEST(XavierStaticFillerTest, TestFillAverage) {
  TypeParam n = (2*4*5 + 1000*4*5) / 2.0;
  this->test_params(FillerParameter_VarianceNorm_AVERAGE, n);
}

template <typename Dtype>
class MSRAFillerTest : public ::testing::Test {
 protected:
  MSRAFillerTest()
      : blob_(new Blob<Dtype>(1000, 2, 4, 5)),
        filler_param_() {
  }
  virtual void test_params(FillerParameter_VarianceNorm variance_norm,
      Dtype n) {
    this->filler_param_.set_variance_norm(variance_norm);
    this->filler_.reset(new MSRAFiller<Dtype>(this->filler_param_));
    this->filler_->Fill(blob_);
    EXPECT_TRUE(this->blob_);
    const int count = this->blob_->count();
    const Dtype* data = this->blob_->cpu_data();
    Dtype mean = 0.;
    Dtype ex2 = 0.;
    for (int i = 0; i < count; ++i) {
      mean += data[i];
      ex2 += data[i] * data[i];
    }
    mean /= count;
    ex2 /= count;
    Dtype std = sqrt(ex2 - mean*mean);
    Dtype target_std = sqrt(2.0 / n);
    EXPECT_NEAR(mean, 0.0, 0.1);
    EXPECT_NEAR(std, target_std, 0.1);
  }
  virtual ~MSRAFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<MSRAFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(MSRAFillerTest, TestDtypes);

TYPED_TEST(MSRAFillerTest, TestFillFanIn) {
  TypeParam n = 2*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_IN, n);
}
TYPED_TEST(MSRAFillerTest, TestFillFanOut) {
  TypeParam n = 1000*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_OUT, n);
}
TYPED_TEST(MSRAFillerTest, TestFillAverage) {
  TypeParam n = (2*4*5 + 1000*4*5) / 2.0;
  this->test_params(FillerParameter_VarianceNorm_AVERAGE, n);
}

template <typename Dtype>
class MSRAStaticFillerTest : public ::testing::Test {
 protected:
  MSRAStaticFillerTest()
      : blob_(new Blob<Dtype>(1000, 2, 4, 5)),
        filler_param_() {
  }
  virtual void test_params(FillerParameter_VarianceNorm variance_norm,
      Dtype n) {
    this->filler_param_.set_variance_norm(variance_norm);
    this->filler_.reset(new MSRAStaticFiller<Dtype>(this->filler_param_));
    this->filler_->Fill(blob_.get());
    EXPECT_TRUE(this->blob_.get());
    const int count = this->blob_->count();
    const Dtype* data = this->blob_->cpu_data();
    Dtype mean = 0.;
    Dtype ex2 = 0.;
    for (int i = 0; i < count; ++i) {
      mean += data[i];
      ex2 += data[i] * data[i];
    }
    mean /= count;
    ex2 /= count;
    Dtype std = sqrt(ex2 - mean*mean);
    Dtype target_std = sqrt(2.0 / n);
    EXPECT_NEAR(mean, 0.0, 0.1);
    EXPECT_NEAR(std, target_std, 0.1);

    // We want to check that repeated calls to the static
    // filler returns the same values. So we copy the first
    // filler call to data_0 and the second one to
    // data_1 and then check whether they are equal.
    std::vector<Dtype> data_0, data_1;
    data_0.resize(count);
    data_1.resize(count);
    caffe_copy(count, data, &data_0.front());

    this->filler_->Fill(this->blob_.get());
    caffe_copy(count, data, &data_1.front());
    for (int i = 0; i < count; ++i) {
      // We do not use EXPECT_FLOAT_EQ because the data must match
      // bit by bit
      EXPECT_EQ(data_0[i], data_1[i]);
    }
  }
  shared_ptr<Blob<Dtype> > const blob_;
  FillerParameter filler_param_;
  shared_ptr<MSRAStaticFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(MSRAStaticFillerTest, TestDtypes);

TYPED_TEST(MSRAStaticFillerTest, TestFillFanIn) {
  TypeParam n = 2*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_IN, n);
}
TYPED_TEST(MSRAStaticFillerTest, TestFillFanOut) {
  TypeParam n = 1000*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_OUT, n);
}
TYPED_TEST(MSRAStaticFillerTest, TestFillAverage) {
  TypeParam n = (2*4*5 + 1000*4*5) / 2.0;
  this->test_params(FillerParameter_VarianceNorm_AVERAGE, n);
}

}  // namespace caffe
