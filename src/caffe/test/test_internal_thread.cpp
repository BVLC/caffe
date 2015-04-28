#include "glog/logging.h"
#include "gtest/gtest.h"

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {


class InternalThreadTest : public ::testing::Test {};

TEST_F(InternalThreadTest, TestStartAndExit) {
  InternalThread thread;
  EXPECT_FALSE(thread.is_started());
  EXPECT_TRUE(thread.StartInternalThread());
  EXPECT_TRUE(thread.is_started());
  EXPECT_TRUE(thread.WaitForInternalThreadToExit());
  EXPECT_FALSE(thread.is_started());
}

class TestThreadA : public InternalThread {
  void InternalThreadEntry() {
    EXPECT_EQ(4244559767, caffe_rng_rand());
  }
};

class TestThreadB : public InternalThread {
  void InternalThreadEntry() {
    EXPECT_EQ(1726478280, caffe_rng_rand());
  }
};

TEST_F(InternalThreadTest, TestRandomSeed) {
  TestThreadA t1;
  Caffe::set_random_seed(9658361);
  EXPECT_TRUE(t1.StartInternalThread());
  EXPECT_TRUE(t1.WaitForInternalThreadToExit());

  TestThreadA t2;
  Caffe::set_random_seed(9658361);
  EXPECT_TRUE(t2.StartInternalThread());
  EXPECT_TRUE(t2.WaitForInternalThreadToExit());

  TestThreadB t3;
  Caffe::set_random_seed(3435563);
  EXPECT_TRUE(t3.StartInternalThread());
  EXPECT_TRUE(t3.WaitForInternalThreadToExit());
}

}  // namespace caffe

