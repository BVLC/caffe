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
  thread.StartInternalThread(Caffe::Get().GetDefaultDevice());
  EXPECT_TRUE(thread.is_started());
  thread.StopInternalThread();
  EXPECT_FALSE(thread.is_started());
}

class TestThreadA : public InternalThread {
  void InternalThreadEntry() {
    if (sizeof(uint_tp) == 4) {
      EXPECT_EQ(2682223724U, caffe_rng_rand());
    } else {
      EXPECT_EQ(10282592414170385089UL, caffe_rng_rand());
    }
  }
};

class TestThreadB : public InternalThread {
  void InternalThreadEntry() {
    if (sizeof(uint_tp) == 4) {
      EXPECT_EQ(887095485U, caffe_rng_rand());
    } else {
      EXPECT_EQ(10310463406559028313UL, caffe_rng_rand());
    }
  }
};

TEST_F(InternalThreadTest, TestRandomSeed) {
  TestThreadA t1;
  Caffe::set_random_seed(9658361, Caffe::GetDefaultDevice());
  t1.StartInternalThread(Caffe::Get().GetDefaultDevice());
  t1.StopInternalThread();

  TestThreadA t2;
  Caffe::set_random_seed(9658361, Caffe::GetDefaultDevice());
  t2.StartInternalThread(Caffe::Get().GetDefaultDevice());
  t2.StopInternalThread();

  TestThreadB t3;
  Caffe::set_random_seed(3435563, Caffe::GetDefaultDevice());
  t3.StartInternalThread(Caffe::Get().GetDefaultDevice());
  t3.StopInternalThread();
}

}  // namespace caffe

