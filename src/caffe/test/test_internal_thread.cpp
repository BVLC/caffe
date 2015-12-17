#include "glog/logging.h"
#include "gtest/gtest.h"

#include "caffe/internal_thread.hpp"

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

}  // namespace caffe

