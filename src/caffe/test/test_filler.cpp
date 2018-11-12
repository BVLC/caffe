#include <vector>

#include "gtest/gtest.h"

#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class ConstantFillerTest : public ::testing::Test {
 protected:
  ConstantFillerTest()
      : blob_(new Blob<Dtype>()),
        filler_param_() {
    filler_param_.set_value(10.);
    filler_.reset(new ConstantFiller<Dtype>(filler_param_));
  }
  virtual void test_params(const vector<int>& shape) {
    EXPECT_TRUE(blob_);
    blob_->Reshape(shape);
    filler_->Fill(blob_);
    const int count = blob_->count();
    const Dtype* data = blob_->cpu_data();
    for (int i = 0; i < count; ++i) {
      EXPECT_EQ(data[i], filler_param_.value());
    }
  }
  virtual ~ConstantFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<ConstantFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(ConstantFillerTest, TestDtypes);

TYPED_TEST(ConstantFillerTest, TestFill) {
  vector<int> blob_shape;
  blob_shape.push_back(2);
  blob_shape.push_back(3);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  this->test_params(blob_shape);
}

TYPED_TEST(ConstantFillerTest, TestFill1D) {
  vector<int> blob_shape(1, 15);
  this->test_params(blob_shape);
}

TYPED_TEST(ConstantFillerTest, TestFill2D) {
  vector<int> blob_shape;
  blob_shape.push_back(8);
  blob_shape.push_back(3);
  this->test_params(blob_shape);
}

TYPED_TEST(ConstantFillerTest, TestFill5D) {
  vector<int> blob_shape;
  blob_shape.push_back(2);
  blob_shape.push_back(3);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  blob_shape.push_back(2);
  this->test_params(blob_shape);
}


template <typename Dtype>
class UniformFillerTest : public ::testing::Test {
 protected:
  UniformFillerTest()
      : blob_(new Blob<Dtype>()),
        filler_param_() {
    filler_param_.set_min(1.);
    filler_param_.set_max(2.);
    filler_.reset(new UniformFiller<Dtype>(filler_param_));
  }
  virtual void test_params(const vector<int>& shape) {
    EXPECT_TRUE(blob_);
    blob_->Reshape(shape);
    filler_->Fill(blob_);
    const int count = blob_->count();
    const Dtype* data = blob_->cpu_data();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], filler_param_.min());
      EXPECT_LE(data[i], filler_param_.max());
    }
  }
  virtual ~UniformFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<UniformFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(UniformFillerTest, TestDtypes);

TYPED_TEST(UniformFillerTest, TestFill) {
  vector<int> blob_shape;
  blob_shape.push_back(2);
  blob_shape.push_back(3);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  this->test_params(blob_shape);
}

TYPED_TEST(UniformFillerTest, TestFill1D) {
  vector<int> blob_shape(1, 15);
  this->test_params(blob_shape);
}

TYPED_TEST(UniformFillerTest, TestFill2D) {
  vector<int> blob_shape;
  blob_shape.push_back(8);
  blob_shape.push_back(3);
  this->test_params(blob_shape);
}

TYPED_TEST(UniformFillerTest, TestFill5D) {
  vector<int> blob_shape;
  blob_shape.push_back(2);
  blob_shape.push_back(3);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  blob_shape.push_back(2);
  this->test_params(blob_shape);
}

template <typename Dtype>
class PositiveUnitballFillerTest : public ::testing::Test {
 protected:
  PositiveUnitballFillerTest()
      : blob_(new Blob<Dtype>()),
        filler_param_() {
    filler_.reset(new PositiveUnitballFiller<Dtype>(filler_param_));
  }
  virtual void test_params(const vector<int>& shape) {
    EXPECT_TRUE(blob_);
    blob_->Reshape(shape);
    filler_->Fill(blob_);
    const int num = blob_->shape(0);
    const int count = blob_->count();
    const int dim = count / num;
    const Dtype* data = blob_->cpu_data();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 0);
      EXPECT_LE(data[i], 1);
    }
    for (int i = 0; i < num; ++i) {
      Dtype sum = Dtype(0);
      for (int j = 0; j < dim; ++j) {
        sum += data[i * dim + j];
      }
      EXPECT_GE(sum, 0.999);
      EXPECT_LE(sum, 1.001);
    }
  }
  virtual ~PositiveUnitballFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<PositiveUnitballFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(PositiveUnitballFillerTest, TestDtypes);

TYPED_TEST(PositiveUnitballFillerTest, TestFill) {
  vector<int> blob_shape;
  blob_shape.push_back(2);
  blob_shape.push_back(3);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  this->test_params(blob_shape);
}

TYPED_TEST(PositiveUnitballFillerTest, TestFill1D) {
  vector<int> blob_shape(1, 15);
  this->test_params(blob_shape);
}

TYPED_TEST(PositiveUnitballFillerTest, TestFill2D) {
  vector<int> blob_shape;
  blob_shape.push_back(8);
  blob_shape.push_back(3);
  this->test_params(blob_shape);
}

TYPED_TEST(PositiveUnitballFillerTest, TestFill5D) {
  vector<int> blob_shape;
  blob_shape.push_back(2);
  blob_shape.push_back(3);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  blob_shape.push_back(2);
  this->test_params(blob_shape);
}

template <typename Dtype>
class GaussianFillerTest : public ::testing::Test {
 protected:
  GaussianFillerTest()
      : blob_(new Blob<Dtype>()),
        filler_param_() {
    filler_param_.set_mean(10.);
    filler_param_.set_std(0.1);
    filler_.reset(new GaussianFiller<Dtype>(filler_param_));
  }
  virtual void test_params(const vector<int>& shape,
      const Dtype tolerance = Dtype(5), const int repetitions = 100) {
    // Tests for statistical properties should be ran multiple times.
    EXPECT_TRUE(blob_);
    blob_->Reshape(shape);
    for (int i = 0; i < repetitions; ++i) {
      test_params_iter(shape, tolerance);
    }
  }
  virtual void test_params_iter(const vector<int>& shape,
      const Dtype tolerance) {
    // This test has a configurable tolerance parameter - by default it was
    // equal to 5.0 which is very loose - allowing some tuning (e.g. for tests
    // on smaller blobs the actual variance will be larger than desired, so the
    // tolerance can be increased to account for that).
    filler_->Fill(blob_);
    const int count = blob_->count();
    const Dtype* data = blob_->cpu_data();
    Dtype mean = Dtype(0);
    Dtype var = Dtype(0);
    for (int i = 0; i < count; ++i) {
      mean += data[i];
      var += data[i] * data[i];
    }
    mean /= count;
    var /= count;
    var -= mean*mean;
    EXPECT_GE(mean, filler_param_.mean() - filler_param_.std() * tolerance);
    EXPECT_LE(mean, filler_param_.mean() + filler_param_.std() * tolerance);
    Dtype target_var = filler_param_.std() * filler_param_.std();
    EXPECT_GE(var, target_var / tolerance);
    EXPECT_LE(var, target_var * tolerance);
  }
  virtual ~GaussianFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<GaussianFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(GaussianFillerTest, TestDtypes);

TYPED_TEST(GaussianFillerTest, TestFill) {
  vector<int> blob_shape;
  blob_shape.push_back(2);
  blob_shape.push_back(3);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  const TypeParam tolerance = TypeParam(3);  // enough for a 120-element blob
  this->test_params(blob_shape, tolerance);
}

TYPED_TEST(GaussianFillerTest, TestFill1D) {
  vector<int> blob_shape(1, 125);
  const TypeParam tolerance = TypeParam(3);
  this->test_params(blob_shape, tolerance);
}

TYPED_TEST(GaussianFillerTest, TestFill2D) {
  vector<int> blob_shape;
  blob_shape.push_back(8);
  blob_shape.push_back(15);
  const TypeParam tolerance = TypeParam(3);
  this->test_params(blob_shape, tolerance);
}

TYPED_TEST(GaussianFillerTest, TestFill5D) {
  vector<int> blob_shape;
  blob_shape.push_back(2);
  blob_shape.push_back(3);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  blob_shape.push_back(2);
  const TypeParam tolerance = TypeParam(2);
  this->test_params(blob_shape, tolerance);
}

template <typename Dtype>
class XavierFillerTest : public ::testing::Test {
 protected:
  XavierFillerTest()
      : blob_(new Blob<Dtype>()),
        filler_param_() {
  }
  virtual void test_params(FillerParameter_VarianceNorm variance_norm,
      Dtype n, const vector<int>& shape, const int repetitions = 100) {
    EXPECT_TRUE(blob_);
    blob_->Reshape(shape);
    for (int i = 0; i < repetitions; ++i) {
      test_params_iter(variance_norm, n);
    }
  }
  virtual void test_params_iter(FillerParameter_VarianceNorm variance_norm,
      Dtype n) {
    filler_param_.set_variance_norm(variance_norm);
    filler_.reset(new XavierFiller<Dtype>(filler_param_));
    filler_->Fill(blob_);
    const int count = blob_->count();
    const Dtype* data = blob_->cpu_data();
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
  vector<int> blob_shape;
  blob_shape.push_back(1000);
  blob_shape.push_back(2);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  TypeParam n = 2*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_IN, n, blob_shape);
}

TYPED_TEST(XavierFillerTest, TestFillFanOut) {
  vector<int> blob_shape;
  blob_shape.push_back(1000);
  blob_shape.push_back(2);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  TypeParam n = 1000*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_OUT, n, blob_shape);
}

TYPED_TEST(XavierFillerTest, TestFillAverage) {
  vector<int> blob_shape;
  blob_shape.push_back(1000);
  blob_shape.push_back(2);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  TypeParam n = (2*4*5 + 1000*4*5) / 2.0;
  this->test_params(FillerParameter_VarianceNorm_AVERAGE, n, blob_shape);
}

TYPED_TEST(XavierFillerTest, TestFill1D) {
  // This makes little sense but at least we will know that we can fill it
  EXPECT_TRUE(this->blob_);
  vector<int> blob_shape(1, 25);
  this->blob_->Reshape(blob_shape);
  this->filler_param_.set_variance_norm(FillerParameter_VarianceNorm_AVERAGE);
  this->filler_.reset(new XavierFiller<TypeParam>(this->filler_param_));
  this->filler_->Fill(this->blob_);
}

TYPED_TEST(XavierFillerTest, TestFill2D) {
  EXPECT_TRUE(this->blob_);
  vector<int> blob_shape;
  blob_shape.push_back(8);
  blob_shape.push_back(3);
  this->blob_->Reshape(blob_shape);
  this->filler_param_.set_variance_norm(FillerParameter_VarianceNorm_AVERAGE);
  this->filler_.reset(new XavierFiller<TypeParam>(this->filler_param_));
  this->filler_->Fill(this->blob_);
}

TYPED_TEST(XavierFillerTest, TestFill5D) {
  EXPECT_TRUE(this->blob_);
  vector<int> blob_shape;
  blob_shape.push_back(2);
  blob_shape.push_back(3);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  blob_shape.push_back(2);
  this->blob_->Reshape(blob_shape);
  this->filler_param_.set_variance_norm(FillerParameter_VarianceNorm_AVERAGE);
  this->filler_.reset(new XavierFiller<TypeParam>(this->filler_param_));
  this->filler_->Fill(this->blob_);
}

template <typename Dtype>
class MSRAFillerTest : public ::testing::Test {
 protected:
  MSRAFillerTest()
      : blob_(new Blob<Dtype>()),
        filler_param_() {
  }
  virtual void test_params(FillerParameter_VarianceNorm variance_norm,
      Dtype n, const vector<int>& shape, const int repetitions = 100) {
    EXPECT_TRUE(blob_);
    blob_->Reshape(shape);
    for (int i = 0; i < repetitions; ++i) {
      test_params_iter(variance_norm, n);
    }
  }
  virtual void test_params_iter(FillerParameter_VarianceNorm variance_norm,
      Dtype n) {
    filler_param_.set_variance_norm(variance_norm);
    filler_.reset(new MSRAFiller<Dtype>(filler_param_));
    filler_->Fill(blob_);
    const int count = blob_->count();
    const Dtype* data = blob_->cpu_data();
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
  vector<int> blob_shape;
  blob_shape.push_back(1000);
  blob_shape.push_back(2);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  TypeParam n = 2*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_IN, n, blob_shape);
}

TYPED_TEST(MSRAFillerTest, TestFillFanOut) {
  vector<int> blob_shape;
  blob_shape.push_back(1000);
  blob_shape.push_back(2);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  TypeParam n = 1000*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_OUT, n, blob_shape);
}

TYPED_TEST(MSRAFillerTest, TestFillAverage) {
  vector<int> blob_shape;
  blob_shape.push_back(1000);
  blob_shape.push_back(2);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  TypeParam n = (2*4*5 + 1000*4*5) / 2.0;
  this->test_params(FillerParameter_VarianceNorm_AVERAGE, n, blob_shape);
}

TYPED_TEST(MSRAFillerTest, TestFill1D) {
  // Like with Xavier - no checking for correctness, just if it can be filled.
  EXPECT_TRUE(this->blob_);
  vector<int> blob_shape(1, 25);
  this->blob_->Reshape(blob_shape);
  this->filler_param_.set_variance_norm(FillerParameter_VarianceNorm_AVERAGE);
  this->filler_.reset(new MSRAFiller<TypeParam>(this->filler_param_));
  this->filler_->Fill(this->blob_);
}

TYPED_TEST(MSRAFillerTest, TestFill2D) {
  EXPECT_TRUE(this->blob_);
  vector<int> blob_shape;
  blob_shape.push_back(8);
  blob_shape.push_back(3);
  this->blob_->Reshape(blob_shape);
  this->filler_param_.set_variance_norm(FillerParameter_VarianceNorm_AVERAGE);
  this->filler_.reset(new MSRAFiller<TypeParam>(this->filler_param_));
  this->filler_->Fill(this->blob_);
}

TYPED_TEST(MSRAFillerTest, TestFill5D) {
  EXPECT_TRUE(this->blob_);
  vector<int> blob_shape;
  blob_shape.push_back(2);
  blob_shape.push_back(3);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  blob_shape.push_back(2);
  this->blob_->Reshape(blob_shape);
  this->filler_param_.set_variance_norm(FillerParameter_VarianceNorm_AVERAGE);
  this->filler_.reset(new MSRAFiller<TypeParam>(this->filler_param_));
  this->filler_->Fill(this->blob_);
}

template <typename Dtype>
class BilinearFillerTest : public ::testing::Test {
 protected:
  BilinearFillerTest()
    : blob_(new Blob<Dtype>()),
      filler_param_() {
  }
  virtual void test_params(const vector<int>& shape) {
    EXPECT_TRUE(blob_);
    blob_->Reshape(shape);
    filler_.reset(new BilinearFiller<Dtype>(filler_param_));
    filler_->Fill(blob_);
    CHECK_EQ(blob_->num_axes(), 4);
    const int outer_num = blob_->count(0, 2);
    const int inner_num = blob_->count(2, 4);
    const Dtype* data = blob_->cpu_data();
    int f = ceil(blob_->shape(3) / 2.);
    Dtype c = (blob_->shape(3) - 1) / (2. * f);
    for (int i = 0; i < outer_num; ++i) {
      for (int j = 0; j < inner_num; ++j) {
        Dtype x = j % blob_->shape(3);
        Dtype y = (j / blob_->shape(3)) % blob_->shape(2);
        Dtype expected_value = (1 - fabs(x / f - c)) * (1 - fabs(y / f - c));
        const Dtype actual_value = data[i * inner_num + j];
        EXPECT_NEAR(expected_value, actual_value, 0.01);
      }
    }
  }
  virtual ~BilinearFillerTest() { delete blob_; }
  Blob<Dtype>* blob_;
  FillerParameter filler_param_;
  shared_ptr<BilinearFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(BilinearFillerTest, TestDtypes);

TYPED_TEST(BilinearFillerTest, TestFillOdd) {
  const int n = 7;
  vector<int> blob_shape;
  blob_shape.push_back(1000);
  blob_shape.push_back(2);
  blob_shape.push_back(n);
  blob_shape.push_back(n);
  this->test_params(blob_shape);
}
TYPED_TEST(BilinearFillerTest, TestFillEven) {
  const int n = 6;
  vector<int> blob_shape;
  blob_shape.push_back(1000);
  blob_shape.push_back(2);
  blob_shape.push_back(n);
  blob_shape.push_back(n);
  this->test_params(blob_shape);
}

}  // namespace caffe
