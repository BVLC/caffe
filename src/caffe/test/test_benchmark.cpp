#include <boost/thread.hpp>

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/util/benchmark.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

const float kMillisecondsThreshold = 30;

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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
>>>>>>> pod/device/blob.hpp
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
>>>>>>> pod/caffe-merge
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
>>>>>>> pod/device/blob.hpp
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
>>>>>>> pod/device/blob.hpp
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
>>>>>>> pod/device/blob.hpp
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
>>>>>>> pod/device/blob.hpp
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
>>>>>>> pod-caffe-pod.hpp-merge
  usleep(300 * 1000);
>>>>>>> origin/BVLC/parallel
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
>>>>>>> device-abstraction
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
  EXPECT_GE(timer.MilliSeconds(), 300 - kMillisecondsThreshold);
  EXPECT_LE(timer.MilliSeconds(), 300 + kMillisecondsThreshold);
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
>>>>>>> pod/device/blob.hpp
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
>>>>>>> pod/caffe-merge
  usleep(300 * 1000);
>>>>>>> origin/BVLC/parallel
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
=======
  usleep(300 * 1000);
>>>>>>> origin/BVLC/parallel
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
>>>>>>> device-abstraction
=======
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
  EXPECT_GE(timer.Seconds(), 0.3 - kMillisecondsThreshold / 1000.);
  EXPECT_LE(timer.Seconds(), 0.3 + kMillisecondsThreshold / 1000.);
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
}

}  // namespace caffe
