#include "glog/logging.h"
#include "gtest/gtest.h"

#include "caffe/internal_thread.hpp"
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/util/math_functions.hpp"
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
#include "caffe/util/math_functions.hpp"
=======
=======
>>>>>>> pod/caffe-merge
>>>>>>> origin/BVLC/parallel
=======
#include "caffe/util/math_functions.hpp"
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {


class InternalThreadTest : public ::testing::Test {};

TEST_F(InternalThreadTest, TestStartAndExit) {
  InternalThread thread;
  EXPECT_FALSE(thread.is_started());
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
  thread.StartInternalThread();
  EXPECT_TRUE(thread.is_started());
  thread.StopInternalThread();
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
  t1.StartInternalThread();
  t1.StopInternalThread();

  TestThreadA t2;
  Caffe::set_random_seed(9658361);
  t2.StartInternalThread();
  t2.StopInternalThread();

  TestThreadB t3;
  Caffe::set_random_seed(3435563);
  t3.StartInternalThread();
  t3.StopInternalThread();
}

<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
  EXPECT_TRUE(thread.StartInternalThread());
  EXPECT_TRUE(thread.is_started());
  EXPECT_TRUE(thread.WaitForInternalThreadToExit());
  EXPECT_FALSE(thread.is_started());
}

>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
}  // namespace caffe

