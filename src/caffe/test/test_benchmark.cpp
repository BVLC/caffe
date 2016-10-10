/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
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
  boost::this_thread::sleep(boost::posix_time::milliseconds(300));
  EXPECT_GE(timer.Seconds(), 0.3 - kMillisecondsThreshold / 1000.);
  EXPECT_LE(timer.Seconds(), 0.3 + kMillisecondsThreshold / 1000.);
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
}

}  // namespace caffe
