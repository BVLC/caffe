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

#include <limits>

#include "gtest/gtest.h"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/float_compare.hpp"

namespace caffe {

    class FloatCompareTest : public ::testing::Test {};

    TEST_F(FloatCompareTest, TestCompareFloatsNans) {
        float a = std::nanf(""), b = std::nanf("");
        float epsilon = 1.0e-3f;
        float diff = caffe::floatDiff(a, b, epsilon);
        EXPECT_TRUE(std::isnan(diff));
    }

    TEST_F(FloatCompareTest, TestCompareFloatsFiniteAndNan) {
        float a = std::nanf(""), b = 1.12345f;
        float epsilon = 1.0e-3f;
        float diff = caffe::floatDiff(a, b, epsilon);
        EXPECT_TRUE(std::isnan(diff));
    }

    TEST_F(FloatCompareTest, TestCompareFloatsInfinity) {
        float a = std::numeric_limits<float>::infinity(),
            b = std::numeric_limits<float>::infinity();
        float epsilon = 1.0e-3f;
        float diff = caffe::floatDiff(a, b, epsilon);
        EXPECT_TRUE(std::isnan(diff));
    }

    TEST_F(FloatCompareTest, TestCompareFloatsBigNegative) {
        float a = 10000.f, epsilon = 1.0e-3f;
        float b = boost::math::float_next(boost::math::float_next(a));
        float diff = caffe::floatDiff(a, b, epsilon);
        EXPECT_NEAR(diff, 0.00195313f, 0.00000001f);
    }

    TEST_F(FloatCompareTest, TestCompareFloatsBigPositive) {
        float a = 10000.f, epsilon = 1.0e-3f;
        float b = boost::math::float_next(a);
        EXPECT_EQ(caffe::floatDiff(a, b, epsilon), FP_ZERO);
    }

    TEST_F(FloatCompareTest, TestCompareFloatsSmallPositive) {
        float a = 0.2304f, b = 0.2306f, epsilon = 1.0e-3f;
        EXPECT_EQ(caffe::floatDiff(a, b, epsilon), FP_ZERO);
    }

    TEST_F(FloatCompareTest, TestCompareFloatsSmallNegative) {
        float a = 0.12f, b = 0.121f, epsilon = 1.0e-3f;
        float diff = caffe::floatDiff(a, b, epsilon);
        EXPECT_NEAR(diff, 0.001f, 0.0001f);
    }

    TEST_F(FloatCompareTest, TestCompareFloatsNearZeroDifferentSigns) {
        float a = -0.2304f, b = 0.2314f, epsilon = 1.0e-3f;
        float diff = caffe::floatDiff(a, b, epsilon);
        EXPECT_NEAR(diff, 0.4618, 0.0001f);
    }
    
    TEST_F(FloatCompareTest, TestCompareFloatsDifferentSigns) {
        float a = -1.f, b = 1.f, epsilon = 1.0e-3f;
        float diff = caffe::floatDiff(a, b, epsilon);
        EXPECT_NEAR(diff, 2.f, epsilon);
    }
}  // namespace caffe
