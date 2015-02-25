#include <unistd.h>  // for usleep

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/util/benchmark.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class BenchmarkTest : public MultiDeviceTest<TypeParam> {};

TYPED_TEST_CASE(BenchmarkTest, TestDtypesAndDevices);

TYPED_TEST(BenchmarkTest, TestTimerConstructor) {
  Timer timer;
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_FALSE(timer.has_run_at_least_once());
}

TYPED_TEST(BenchmarkTest, TestTimerStart) {
  Timer timer;
  timer.Start();
  EXPECT_TRUE(timer.initted());
  EXPECT_TRUE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
  timer.Start();
  EXPECT_TRUE(timer.initted());
  EXPECT_TRUE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
  timer.Stop();
  timer.Start();
  EXPECT_TRUE(timer.initted());
  EXPECT_TRUE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
}

TYPED_TEST(BenchmarkTest, TestTimerStop) {
  Timer timer;
  timer.Stop();
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_FALSE(timer.has_run_at_least_once());
  timer.Start();
  timer.Stop();
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
  timer.Stop();
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
}

TYPED_TEST(BenchmarkTest, TestTimerMilliSeconds) {
  Timer timer;
  EXPECT_EQ(timer.MilliSeconds(), 0);
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_FALSE(timer.has_run_at_least_once());
  timer.Start();
  usleep(300 * 1000);
  EXPECT_GE(timer.MilliSeconds(), 290);
  EXPECT_LE(timer.MilliSeconds(), 310);
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
}

TYPED_TEST(BenchmarkTest, TestTimerSeconds) {
  Timer timer;
  EXPECT_EQ(timer.Seconds(), 0);
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_FALSE(timer.has_run_at_least_once());
  timer.Start();
  usleep(300 * 1000);
  EXPECT_GE(timer.Seconds(), 0.290);
  EXPECT_LE(timer.Seconds(), 0.310);
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
}

}  // namespace caffe
