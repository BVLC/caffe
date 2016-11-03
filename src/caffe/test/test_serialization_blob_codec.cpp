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

#include <boost/assign.hpp>
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include "caffe/internode/configuration.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
namespace {

using ::testing::_;
using ::testing::Return;

template <typename TypeParam>
class BlobCodecTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
};
TYPED_TEST_CASE(BlobCodecTest, TestDtypesAndDevices);

TYPED_TEST(BlobCodecTest, encode_4321_diff) {
  BlobUpdate msg;
  Blob<float> srcblob;
  vector<int> v = boost::assign::list_of(4)(3)(2)(1);
  srcblob.Reshape(v);
  vector<float> diff = boost::assign::list_of(999.99)(12.3)(0.1)(-3.3)
                                             (+2.0)(12.3)(10.2)(FLT_MAX)
                                             (+4.4)(12.3)(0.0)(-1.3)
                                             (+6.5)(12.3)(24.42)(1010.10)
                                             (FLT_MIN)(12.3)(66.6)(133.1)
                                             (12.4)(12.3)(0.0001)(100.3);

  ASSERT_EQ(diff.size(), srcblob.count());

  caffe_copy(srcblob.count(),
          &diff.front(),
          srcblob.mutable_cpu_diff());

  shared_ptr<BlobCodec<float> > codec = BlobCodec<float>::create_codec(
          MultinodeParameter::default_instance(), true);

  codec->encode(&msg, &srcblob, BlobEncoding::GRADS, msg.info().part());

  EXPECT_EQ(0, memcmp(msg.data().c_str(), &diff.front(),
           sizeof(float)*diff.size()));
}

TYPED_TEST(BlobCodecTest, encode_decode_2222_data) {
  BlobUpdate msg;
  Blob<float> srcblob;
  Blob<float> dstblob;
  vector<int> v = boost::assign::list_of(2)(2)(2)(2);
  srcblob.Reshape(v);
  dstblob.Reshape(v);
  vector<float> data = boost::assign::list_of(1.1)(-2.2)(3.3)(5.5)
                                             (6.6)(-7.7)(8.8)(9.9)
                                             (13.13)(-12.12)(12.12)(11.11)
                                             (128.128)(-132.312)(1.1)(-10.10);
  vector<float> data_zero = boost::assign::list_of(0.0)(0.0)(0.0)(0.0)
                                                  (0.0)(0.0)(0.0)(0.0)
                                                  (0.0)(0.0)(0.0)(0.0)
                                                  (0.0)(0.0)(0.0)(0.0);
  ASSERT_EQ(data.size(), srcblob.count());
  ASSERT_EQ(data_zero.size(), dstblob.count());

  caffe_copy<float>(srcblob.count(),
          &data.front(),
          srcblob.mutable_cpu_data());
  caffe_copy<float>(dstblob.count(),
          &data_zero.front(),
          dstblob.mutable_cpu_diff());

  shared_ptr<BlobCodec<float> > codec = BlobCodec<float>::create_codec(
          MultinodeParameter::default_instance(), true);

  codec->encode(&msg, &srcblob, BlobEncoding::PARAMS, msg.info().part());
  codec->decode(msg,  &dstblob, BlobEncoding::PARAMS, 1.0f, 0.0f);

  EXPECT_EQ(0, memcmp(dstblob.cpu_data(), &data.front(),
          sizeof(float)*dstblob.count()));
}

TYPED_TEST(BlobCodecTest, encode_8width_data_) {
  BlobUpdate msg;
  Blob<float> srcblob;
  vector<int> v = boost::assign::list_of(1)(1)(1)(8);
  srcblob.Reshape(v);
  vector<float> data = boost::assign::list_of(-0.0)(-0.3)(-2.2)(-3.3)
                                             (+0.0)(12.3)(10.2)(-1.3);

  ASSERT_EQ(data.size(), srcblob.count());

  caffe_copy<float>(srcblob.count(),
          &data.front(),
          srcblob.mutable_cpu_data());

  shared_ptr<BlobCodec<float> > codec = BlobCodec<float>::create_codec(
          MultinodeParameter::default_instance(), true);

  codec->encode(&msg, &srcblob, BlobEncoding::PARAMS, msg.info().part());

  EXPECT_EQ(0, memcmp(msg.data().c_str(), &data.front(),
          sizeof(float)*data.size()));
}

TYPED_TEST(BlobCodecTest, encode_4width_diff) {
  BlobUpdate msg;
  Blob<float> srcblob;
  vector<int> v = boost::assign::list_of(1)(1)(1)(4);
  srcblob.Reshape(v);
  vector<float> diff = boost::assign::list_of(-0.0)(-99.99)(-0.3)(0.4);

  ASSERT_EQ(diff.size(), srcblob.count());

  caffe_copy<float>(srcblob.count(),
          &diff.front(),
          srcblob.mutable_cpu_diff());

  shared_ptr<BlobCodec<float> > codec = BlobCodec<float>::create_codec(
          MultinodeParameter::default_instance(), true);

  codec->encode(&msg, &srcblob, BlobEncoding::GRADS, msg.info().part());

  EXPECT_EQ(0, memcmp(msg.data().c_str(), &diff.front(),
          sizeof(float)*diff.size()));
}

TYPED_TEST(BlobCodecTest, encode_decode_4width_diff) {
  BlobUpdate msg;
  Blob<float> srcblob;
  Blob<float> dstblob;
  vector<int> v = boost::assign::list_of(1)(1)(1)(4);
  srcblob.Reshape(v);
  dstblob.Reshape(v);
  vector<float> diff_zero = boost::assign::list_of(0.0)(0.0)(0.0)(0.0);
  vector<float> diff = boost::assign::list_of(1.0)(2.2)(3.3)(4.4);

  ASSERT_EQ(diff.size(), srcblob.count());
  ASSERT_EQ(diff_zero.size(), srcblob.count());

  caffe_copy<float>(srcblob.count(),
          &diff.front(),
          srcblob.mutable_cpu_diff());
  caffe_copy<float>(dstblob.count(),
          &diff_zero.front(),
          dstblob.mutable_cpu_diff());

  shared_ptr<BlobCodec<float> > codec = BlobCodec<float>::create_codec(
           MultinodeParameter::default_instance(), true);

  codec->encode(&msg, &srcblob, BlobEncoding::GRADS, msg.info().part());
  codec->decode(msg,  &dstblob, BlobEncoding::GRADS, 1.0f, 0.0f);

  EXPECT_EQ(0, memcmp(dstblob.cpu_diff(), &diff.front(),
          sizeof(float)*dstblob.count()));
}

TYPED_TEST(BlobCodecTest, encode_decode_4width_data) {
  BlobUpdate msg;
  Blob<float> srcblob;
  Blob<float> dstblob;
  vector<int> v = boost::assign::list_of(1)(1)(1)(4);
  srcblob.Reshape(v);
  dstblob.Reshape(v);
  vector<float> data_zero = boost::assign::list_of(0.0)(0.0)(0.0)(0.0);
  vector<float> data = boost::assign::list_of(4.0)(3.2)(2.3)(1.4);

  ASSERT_EQ(data.size(), srcblob.count());
  ASSERT_EQ(data_zero.size(), srcblob.count());

  caffe_copy<float>(srcblob.count(),
          &data.front(),
          srcblob.mutable_cpu_data());
  caffe_copy<float>(dstblob.count(),
          &data_zero.front(),
          dstblob.mutable_cpu_data());

  shared_ptr<BlobCodec<float> > codec = BlobCodec<float>::create_codec(
          MultinodeParameter::default_instance(), true);

  codec->encode(&msg, &srcblob, BlobEncoding::PARAMS, msg.info().part());
  codec->decode(msg,  &dstblob, BlobEncoding::PARAMS, 1.0f, 0.0f);

  EXPECT_EQ(0, memcmp(dstblob.cpu_data(), &data.front(),
          sizeof(float)*dstblob.count()));
}

TYPED_TEST(BlobCodecTest, encode_decode_4width_data_alpha_0_5) {
  BlobUpdate msg;
  Blob<float> srcblob;
  Blob<float> dstblob;
  vector<int> v = boost::assign::list_of(1)(1)(1)(4);
  srcblob.Reshape(v);
  dstblob.Reshape(v);
  vector<float> data_zero = boost::assign::list_of(0.0)(0.0)(0.0)(0.0);
  vector<float> data = boost::assign::list_of(4.0)(3.2)(2.4)(1.4);
  vector<float> data_expected = boost::assign::list_of(2.0)(1.6)(1.2)(0.7);

  ASSERT_EQ(data.size(), srcblob.count());
  ASSERT_EQ(data_zero.size(), srcblob.count());

  caffe_copy<float>(srcblob.count(),
          &data.front(),
          srcblob.mutable_cpu_data());
  caffe_copy<float>(dstblob.count(),
          &data_zero.front(),
          dstblob.mutable_cpu_data());

  shared_ptr<BlobCodec<float> > codec = BlobCodec<float>::create_codec(
          MultinodeParameter::default_instance(), true);

  codec->encode(&msg, &srcblob, BlobEncoding::PARAMS, msg.info().part());
  codec->decode(msg,  &dstblob, BlobEncoding::PARAMS, 0.5f, 0.0f);

  EXPECT_EQ(0, memcmp(dstblob.cpu_data(), &data_expected.front(),
          sizeof(float)*dstblob.count()));
}

TYPED_TEST(BlobCodecTest, encode_decode_4width_data_beta_0_5) {
  BlobUpdate msg;
  Blob<float> srcblob;
  Blob<float> dstblob;
  vector<int> v = boost::assign::list_of(1)(1)(1)(4);
  srcblob.Reshape(v);
  dstblob.Reshape(v);
  vector<float> data_one = boost::assign::list_of(1.0)(1.0)(1.0)(1.0);
  vector<float> data = boost::assign::list_of(4.0)(3.2)(2.4)(1.4);
  vector<float> data_expected = boost::assign::list_of(4.5)(3.7)(2.9)(1.9);

  ASSERT_EQ(data.size(), srcblob.count());
  ASSERT_EQ(data_one.size(), srcblob.count());

  caffe_copy<float>(srcblob.count(), &data.front(),
          srcblob.mutable_cpu_data());
  caffe_copy<float>(dstblob.count(), &data_one.front(),
          dstblob.mutable_cpu_data());

  shared_ptr<BlobCodec<float> > codec = BlobCodec<float>::create_codec(
          MultinodeParameter::default_instance(), true);

  codec->encode(&msg, &srcblob, BlobEncoding::PARAMS, msg.info().part());
  codec->decode(msg,  &dstblob, BlobEncoding::PARAMS, 1.0f, 0.5f);

  EXPECT_EQ(0, memcmp(dstblob.cpu_data(), &data_expected.front(),
          sizeof(float)*dstblob.count()));
}

TYPED_TEST(BlobCodecTest, encode_decode_4width_data_alpha_0_5_beta_0_5) {
  BlobUpdate msg;
  Blob<float> srcblob;
  Blob<float> dstblob;
  vector<int> v = boost::assign::list_of(1)(1)(1)(4);
  srcblob.Reshape(v);
  dstblob.Reshape(v);
  vector<float> data_one = boost::assign::list_of(1.0)(1.0)(1.0)(1.0);
  vector<float> data = boost::assign::list_of(4.0)(3.2)(2.4)(1.4);
  vector<float> data_expected = boost::assign::list_of(2.5)(2.1)(1.7)(1.2);

  ASSERT_EQ(data.size(), srcblob.count());
  ASSERT_EQ(data_one.size(), srcblob.count());

  caffe_copy<float>(srcblob.count(), &data.front(),
          srcblob.mutable_cpu_data());
  caffe_copy<float>(dstblob.count(), &data_one.front(),
          dstblob.mutable_cpu_data());

  shared_ptr<BlobCodec<float> > codec = BlobCodec<float>::create_codec(
          MultinodeParameter::default_instance(), true);

  codec->encode(&msg, &srcblob, BlobEncoding::PARAMS, msg.info().part());
  codec->decode(msg,  &dstblob, BlobEncoding::PARAMS, 0.5f, 0.5f);

  EXPECT_EQ(0, memcmp(dstblob.cpu_data(), &data_expected.front(),
          sizeof(float)*dstblob.count()));
}

TYPED_TEST(BlobCodecTest, encode_decode_4width_data_alpha_0_beta_1) {
  BlobUpdate msg;
  Blob<float> srcblob;
  Blob<float> dstblob;
  vector<int> v = boost::assign::list_of(1)(1)(1)(4);
  srcblob.Reshape(v);
  dstblob.Reshape(v);
  vector<float> data_one = boost::assign::list_of(1.1)(2.3)(1.4)(0.01);
  vector<float> data = boost::assign::list_of(4.0)(3.2)(2.4)(1.4);
  vector<float> data_expected = boost::assign::list_of(1.1)(2.3)(1.4)(0.01);

  ASSERT_EQ(data.size(), srcblob.count());
  ASSERT_EQ(data_one.size(), srcblob.count());

  caffe_copy<float>(srcblob.count(), &data.front(),
          srcblob.mutable_cpu_data());
  caffe_copy<float>(dstblob.count(), &data_one.front(),
          dstblob.mutable_cpu_data());

  shared_ptr<BlobCodec<float> > codec = BlobCodec<float>::create_codec(
          MultinodeParameter::default_instance(), true);

  codec->encode(&msg, &srcblob, BlobEncoding::PARAMS, msg.info().part());
  codec->decode(msg,  &dstblob, BlobEncoding::PARAMS, 0.0f, 1.0f);

  EXPECT_EQ(0, memcmp(dstblob.cpu_data(), &data_expected.front(),
          sizeof(float)*dstblob.count()));
}


}  // namespace
}  // namespace caffe
