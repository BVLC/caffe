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
#include <boost/make_shared.hpp>
#include <caffe/test/test_caffe_main.hpp>
#include <gmock/gmock.h>
#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/internode/communication.hpp"
#include "caffe/internode/configuration.hpp"
#include "caffe/internode/tree_cluster.hpp"
#include "caffe/multinode/BlobComms.hpp"

namespace caffe {
namespace {

using ::testing::_;
using ::testing::Return;
using ::testing::AtLeast;
using ::testing::Test;
using ::testing::StrictMock;
using ::testing::NiceMock;
using ::testing::InSequence;
using ::testing::Mock;
using ::testing::SaveArg;
using ::testing::Invoke;
using ::boost::assign::list_of;
using ::caffe::internode::Waypoint;

struct BlobConstInfoMock : BlobConstInfo {
  MOCK_CONST_METHOD2(parts, uint32_t(int a, int b));
  MOCK_CONST_METHOD1(parts, uint32_t(int a));
  MOCK_CONST_METHOD0(parts, uint32_t());
  MOCK_CONST_METHOD1(blobs, uint32_t(int a));
  MOCK_CONST_METHOD0(layers, uint32_t());
  MOCK_CONST_METHOD1(needs_syncing, bool(int a));
};

struct BlobSyncInfoMock : BlobSyncInfo {
  MOCK_METHOD5(received, bool(
          internode::RemoteId from,
          int layer_id, int blob_id, int part, uint32_t version));

  MOCK_METHOD1(register_synced_handler, void(BlobSyncInfo::Handler* handler));
  MOCK_CONST_METHOD4(received_version, uint32_t(
          internode::RemoteId from,
          int layer_id, int blob_id, int part));

  MOCK_METHOD1(add_remote, void(internode::RemoteId id));
  MOCK_METHOD1(remove_remote, void(internode::RemoteId id));
};

struct WaypointMock : internode::Waypoint {
  MOCK_METHOD3(async_send, void(const char* buffer, size_t size, SentCallback));
  MOCK_CONST_METHOD0(id, internode::RemoteId());
  MOCK_CONST_METHOD0(address, string());
  MOCK_METHOD1(register_receive_handler, void(Waypoint::Handler* handler));
  MOCK_CONST_METHOD0(guaranteed_comm, bool());
  MOCK_CONST_METHOD0(max_packet_size, size_t());
};

template<typename Dtype>
struct BlobKeyChainMock : public BlobKeyChain<Dtype> {
  MOCK_METHOD1(lock, void(int layer_id));
  MOCK_METHOD3(lock, void(int layer_id, int blob_id, int part));
  MOCK_METHOD1(unlock, void(int layer_id));
  MOCK_METHOD3(unlock, void(int layer_id, int blob_id, int part));
};

template <typename Dtype>
class BlobCodecMock : public BlobCodec<Dtype> {
 public:
     typedef typename BlobEncoding::What What;
//     template <Dtype>
     BlobCodecMock() {
      real_ = BlobCodec<Dtype>::create_codec(
         MultinodeParameter::default_instance(), true);
      ON_CALL(*this, encode(_, _, _, _))
        .WillByDefault(Invoke(real_.get(), &BlobCodec<Dtype>::encode));
      ON_CALL(*this, decode(_, _, _, _, _))
        .WillByDefault(Invoke(real_.get(), &BlobCodec<Dtype>::decode));
      ON_CALL(*this, max_elements_per_part())
        .WillByDefault(Invoke(real_.get(),
              &BlobCodec<Dtype>::max_elements_per_part));
      ON_CALL(*this, packet_size())
        .WillByDefault(Invoke(real_.get(), &BlobCodec<Dtype>::packet_size));
     }
    uint32_t encode_real(BlobUpdate* msg,
                            const Blob<Dtype>* src,
                            What what,
                            uint32_t part) const {
        return real_->encode(msg, src, what, part);
    }
    MOCK_CONST_METHOD4_T(encode, uint32_t(BlobUpdate* msg,
                          const Blob<Dtype>* src,
                          What what,
                          uint32_t part));

    MOCK_CONST_METHOD5_T(decode, bool(const BlobUpdate& update,
                      Blob<Dtype>* dest,
                      What what,
                      Dtype alpha,
                      Dtype beta));

    MOCK_CONST_METHOD0_T(max_elements_per_part, size_t());
    MOCK_CONST_METHOD0_T(packet_size, size_t());

 private:
  shared_ptr<BlobCodec<Dtype> > real_;
};

template<typename Dtype>
class BlobAccessorTestImpl : public BlobAccessor<Dtype> {
  vector<int> v;
 public:
    Blob<Dtype> dummy_blob;
    BlobAccessorTestImpl()
  : v(boost::assign::list_of(1).operator vector<int> ())
  , dummy_blob(v) {}

  virtual Blob<Dtype>* get_blob(int layer, int blob_id) {
    return &dummy_blob;
  }
};

template <typename Dtype>
class IterSizeHandlerMock: public BlobComms<Dtype>::IterSizeHandler {
 public:
    MOCK_METHOD2(received_iter_size, void(internode::RemoteId from, int iters));
};

template <typename Dtype>
class BlobAccessorMock : public BlobAccessorTestImpl<Dtype> {
 public:
//    typedef typename BlobEncoding::What What;
//     template <Dtype>
    BlobAccessorMock() {
        real_.reset( new BlobAccessorTestImpl<float>());
        ON_CALL(*this, get_blob(_, _))
                .WillByDefault(Invoke(real_.get(),
                                      &BlobAccessorTestImpl<Dtype>::get_blob));
    }
    MOCK_METHOD2_T(get_blob, Blob<Dtype>*(int layer, int blob_id));

 private:
    shared_ptr<BlobAccessorTestImpl<Dtype> > real_;
};

MATCHER_P(BufferEq, str, "") {
  return std::equal(str.c_str(), str.c_str() + str.size(), arg);
}

MATCHER_P4(BlobUpdateInfoEq, layer_id, blob_id, part, version, "") {
  return  arg->info().layer_id() == layer_id
          && arg->info().blob_id() == blob_id
          && arg->info().part() == part
          && arg->info().version() == version;
}
MATCHER_P4(BlobUpdateInfoEqRef, layer_id, blob_id, part, version, "") {
  return  arg.info().layer_id() == layer_id
          && arg.info().blob_id() == blob_id
          && arg.info().part() == part
          && arg.info().version() == version;
}

template <typename TypeParam>
class BlobCommsTest : public MultiDeviceTest<TypeParam> {
 public:
  shared_ptr<BlobCodecMock<float> > codec_mock;
  shared_ptr<WaypointMock> waypoint_mock;
  shared_ptr<IterSizeHandlerMock<float> > iter_size_handler_mock1;
  shared_ptr<IterSizeHandlerMock<float> > iter_size_handler_mock2;

  shared_ptr<BlobAccessorMock<float> > blob_accessor_mock;
  shared_ptr<BlobConstInfoMock> const_info_mock;
  shared_ptr<BlobSyncInfoMock> sync_info_mock;
  shared_ptr<BlobKeyChain<float> > keychain;
  shared_ptr<BlobKeyChainMock<float> > keychain_mock;
  shared_ptr<BlobComms<float> > comms;
  BlobComms<float>::Settings settings;
  Waypoint::SentCallback callback;

  void prepare_const_mock(vector<vector<int> > vparts,
                          bool register_handler = true) {
    EXPECT_CALL(*const_info_mock, layers())
            .WillRepeatedly(Return(vparts.size()));
    for (int i = 0; i < vparts.size(); ++i) {
      EXPECT_CALL(*const_info_mock, blobs(i))
              .WillRepeatedly(Return(vparts.at(i).size()));
      int totalpartsinlayer = 0;
      for (int j = 0; j < vparts.at(i).size(); ++j) {
        EXPECT_CALL(*const_info_mock, parts(i, j))
              .WillRepeatedly(Return(vparts.at(i).at(j)));
        totalpartsinlayer+=vparts.at(i).at(j);
      }
      EXPECT_CALL(*const_info_mock, parts(i))
              .WillRepeatedly(Return(totalpartsinlayer));
    }
    EXPECT_CALL(*const_info_mock, needs_syncing(_))
            .WillRepeatedly(Return(true));
  }

  virtual void SetUp() {
    waypoint_mock.reset(new StrictMock<WaypointMock>());
    codec_mock.reset(new NiceMock<BlobCodecMock<float> >());
    const_info_mock.reset(new StrictMock<BlobConstInfoMock>());
    iter_size_handler_mock1.reset(
        new StrictMock<IterSizeHandlerMock<float> >());
    iter_size_handler_mock2.reset(
        new StrictMock<IterSizeHandlerMock<float> >());

    prepare_const_mock(
        list_of<vector<int> >
            (list_of<int>(1)(1)(1)(1)(1)(1))    // layer0
            (list_of<int>(1)(1)(1)(1)(1)(1))    // layer1
            (list_of<int>(1)(1)(1)(1)(1)(1))    // layer2
            (list_of<int>(1)(1)(1)(1)(1)(1)));  // layer3

    keychain_mock.reset(new StrictMock<BlobKeyChainMock<float> >() );

    blob_accessor_mock.reset(new NiceMock<BlobAccessorMock<float> >());
    sync_info_mock.reset(new StrictMock<BlobSyncInfoMock>());
  }

  BlobCommsTest()
      : settings(BlobComms<float>::Settings(
              BlobEncoding::GRADS, BlobEncoding::PARAMS, 1.0, 0.0)) {}

  virtual void TearDown() {
    comms.reset();
  }

  void buildSendMethodExpects(
          int layer_id,
          int blob_id,
          int part_id,
          uint32_t version,
          Waypoint::SentCallback *callback,
          int times) {
    BlobUpdate update;
    update.mutable_info()->set_layer_id(layer_id);
    update.mutable_info()->set_blob_id(blob_id);
    update.mutable_info()->set_part(part_id);
    update.mutable_info()->set_version(version);
    buildSendMethodExpectations(layer_id, blob_id, part_id,
          version, callback, &update, times);
  }

  void buildSendMethodExpectations(
          int layer_id,
          int blob_id,
          int part_id,
          uint32_t version,
          Waypoint::SentCallback* callback,
          BlobUpdate *update,
          int times) {
    codec_mock->encode_real(
        update, &blob_accessor_mock->dummy_blob,
        settings.what_sent, update->info().part());

    string str = update->SerializeAsString();
    {
      InSequence dummy;
      for (int i = 0; i < times; ++i) {
        EXPECT_CALL(*keychain_mock,
                    lock(layer_id));
        EXPECT_CALL(*blob_accessor_mock, get_blob(layer_id, blob_id));
        EXPECT_CALL(*codec_mock,
                    encode(
                        BlobUpdateInfoEq(layer_id, blob_id, part_id, version),
                        _,
                        BlobEncoding::GRADS, part_id));
        EXPECT_CALL(*keychain_mock,
                    unlock(layer_id));
        EXPECT_CALL(*waypoint_mock,
                    async_send(BufferEq(str), update->ByteSize(), _))
            .WillOnce(SaveArg<2>(callback));
      }
    }
  }

  void SendIterSize(
          shared_ptr<WaypointMock> waypoint_mock,
          int size) {
    BlobUpdate update;
    update.set_iters(size);
    string str = update.SerializeAsString();
    EXPECT_CALL(*waypoint_mock, async_send(BufferEq(str), update.ByteSize(), _))
            .WillOnce(SaveArg<2>(&callback));
    comms->send_iter_size(size);
    callback(true);
  }

  void buildOne() {
    keychain = BlobKeyChain<float>::create_empty(const_info_mock->layers());
    comms = BlobComms<float>::create(blob_accessor_mock,
            const_info_mock, sync_info_mock, waypoint_mock, codec_mock,
            keychain_mock, settings, 1);
  }
};

TYPED_TEST_CASE(BlobCommsTest, TestDtypesAndDevices);

TYPED_TEST(BlobCommsTest, SendIterSize) {
    this->buildOne();
    this->SendIterSize(this->waypoint_mock, 10);
    this->SendIterSize(this->waypoint_mock, -1);
    this->SendIterSize(this->waypoint_mock, 0);
    this->SendIterSize(this->waypoint_mock, 101);
    this->SendIterSize(this->waypoint_mock, 1);
}

TYPED_TEST(BlobCommsTest, pushOneWithCancelledVersion) {
  this->buildOne();
  int layer_id = 0, blob_id = 0, part_id = 0, version = 1;
  int times = 0;

  this->buildSendMethodExpects(layer_id, blob_id, part_id,
                 version, NULL, times);
  this->comms->cancel(layer_id, version);
  this->comms->push(layer_id, blob_id, part_id, version);
}

TYPED_TEST(BlobCommsTest, pushOne) {
  this->buildOne();
  int layer_id = 0, blob_id = 0, part_id = 0, version = 1;
  int times = 1;
  this->buildSendMethodExpects(layer_id, blob_id, part_id,
                             version, &this->callback, times);
  this->comms->push(layer_id, blob_id, part_id, version);
  this->callback(true);
}

TYPED_TEST(BlobCommsTest, pushAnotherTwoDuringSending) {
  this->buildOne();
  int layer_id = 0, blob_id = 0, part_id = 0, version = 1;
  int times = 1;
  this->buildSendMethodExpects(layer_id, blob_id, part_id,
                            version, &this->callback, times);
  this->comms->push(layer_id, blob_id, part_id, version);
  this->comms->push(layer_id, blob_id, part_id, version);
  this->comms->push(layer_id, blob_id, part_id, version);
}

TYPED_TEST(BlobCommsTest, push3OneByOne) {
  this->buildOne();
  int layer_id = 0, blob_id = 0, part_id = 0, version = 1;
  int times = 3;

  this->buildSendMethodExpects(layer_id, blob_id, part_id,
                             version, &this->callback, times);
  this->comms->push(layer_id, blob_id, part_id, version);
  // simulate Waypoint async_send => implicit call BlobComms::sent()
  // clears during_sending BlobComms state
  this->callback(true);
  this->comms->push(layer_id, blob_id, part_id, version);
  this->callback(true);
  this->comms->push(layer_id, blob_id, part_id, version);
  this->callback(true);
}

TYPED_TEST(BlobCommsTest, cancelOneWhenInQueueDuringSending3Queue) {
  this->buildOne();
  int layer_id = 0, blob_id = 0, part_id = 0, version = 1;
  int times = 2;

  this->buildSendMethodExpects(layer_id, blob_id, part_id,
                   version, &this->callback, times);
  this->comms->push(layer_id, blob_id, part_id, version);
  // simulate Waypoint async_send => implicit call BlobComms::sent()
  // clears during_sending BlobComms state
  this->callback(true);
  this->comms->push(layer_id, blob_id, part_id, version);
  this->callback(true);
  this->comms->cancel(layer_id, version);
  this->comms->push(layer_id, blob_id, part_id, version);
  this->callback(true);
}

TYPED_TEST(BlobCommsTest, cancelLayer1WhenInQueue) {
//  BlobCommsBase bb ;
  this->buildOne();
  int blob_id = 0, part_id = 0, version = 1;
  {
    InSequence dummy;

    this->buildSendMethodExpects(2, blob_id, part_id, 2, &this->callback, 1);
    this->buildSendMethodExpects(2, blob_id, part_id, 3, &this->callback, 1);
    this->buildSendMethodExpects(3, blob_id, part_id,
                                         version, &this->callback, 1);
  }
    EXPECT_CALL(*this->keychain_mock, lock(1)).Times(0);
    EXPECT_CALL(*this->keychain_mock, unlock(1)).Times(0);
    EXPECT_CALL(*this->blob_accessor_mock, get_blob(1, _)).Times(0);

    this->comms->cancel(1, version);
    this->comms->push(2, blob_id, part_id, 2);
    this->comms->push(3, blob_id, part_id, version);
    this->comms->push(1, blob_id, part_id, version);
    this->comms->push(2, blob_id, part_id, 3);


  this->callback(true);
  this->callback(true);
  this->callback(true);
}

TYPED_TEST(BlobCommsTest, pushParamsOutOfRange) {
    this->buildOne();
    int part_id = 0;

    EXPECT_DEATH(this->comms->push(1, 33, part_id, 1), "");
    EXPECT_DEATH(this->comms->push(1,
                 this->const_info_mock->blobs(1), part_id, 1), "");
    EXPECT_DEATH(this->comms->push(this->const_info_mock->layers(),
                 0, part_id, 1), "");
    EXPECT_DEATH(this->comms->push(1, 0, 45, 1), "");
}

TYPED_TEST(BlobCommsTest, checkPriorityQueue) {
  this->buildOne();
  int part_id = 0;
  {
    InSequence dummy;
    this->buildSendMethodExpects(2,  0, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(2,  4, part_id, 5, &this->callback, 1);
    this->buildSendMethodExpects(1,  0, part_id, 5, &this->callback, 1);
    this->buildSendMethodExpects(1,  1, part_id, 5, &this->callback, 1);
    this->buildSendMethodExpects(3,  5, part_id, 5, &this->callback, 1);
    this->buildSendMethodExpects(3,  4, part_id, 5, &this->callback, 1);
    this->buildSendMethodExpects(3,  3, part_id, 5, &this->callback, 1);
    this->buildSendMethodExpects(3,  2, part_id, 5, &this->callback, 1);
    this->buildSendMethodExpects(3,  1, part_id, 5, &this->callback, 1);
    this->buildSendMethodExpects(1,  4, part_id, 5, &this->callback, 1);
    this->buildSendMethodExpects(1,  2, part_id, 5, &this->callback, 1);
    this->buildSendMethodExpects(1,  3, part_id, 5, &this->callback, 1);
    this->buildSendMethodExpects(2,  3, part_id, 5, &this->callback, 1);
    this->buildSendMethodExpects(2,  2, part_id, 5, &this->callback, 1);
    this->buildSendMethodExpects(2,  1, part_id, 5, &this->callback, 1);
  }

  // queue 2:[5,4,3,2,1]
  this->comms->push(2, 0, part_id, 1);  //  2v1 sent immediately
  this->comms->push(2, 1, part_id, 2);  // then hold bof during_sending state
  this->comms->push(2, 2, part_id, 3);
  this->comms->push(2, 3, part_id, 4);
  this->comms->push(2, 4, part_id, 5);
  // callback immitates incoming data and turns 'during sending' state to
  // 'ready' for sending
  this->callback(true);  // sent 2v5 => [2:[4,3,2]]
  // queue 1:[5,4,3,2,1]
  this->comms->push(1, 3, part_id, 1);  // holds 2v4
  this->comms->push(1, 2, part_id, 2);
  this->comms->push(1, 4, part_id, 3);
  this->comms->push(1, 1, part_id, 4);
  this->comms->push(1, 0, part_id, 5);

  this->callback(true);  // sent 1v5 => [1:[4,3,2,1], 2:[4,3,2]]
  this->callback(true);  // sent 1v4 as 1v5 (sending version determines args)
//    =>[1:[3,2,1], 2:[4,3,2]]
  // queue 3:[5,4,3,2,1]
  this->comms->push(3, 1, part_id, 1);  // holds 1v3
  this->comms->push(3, 2, part_id, 2);
  this->comms->push(3, 3, part_id, 3);
  this->comms->push(3, 4, part_id, 4);
  this->comms->push(3, 5, part_id, 5);
//    => [3:[5,4,3,2,1], 1:[3,2,1],2:[4,3,2]]

  this->callback(true);   // sent 3v5 => [3:[4,3,2,1], 1:[3,2,1],2:[4,3,2]]
  this->callback(true);   // sent 3v4 as 3v5 => [3:[3,2,1], 1:[3,2,1],2:[4,3,2]]
  this->callback(true);   // sent 3v3 as 3v5 => [3:[2,1], 1:[3,2,1],2:[4,3,2]]
  this->callback(true);   // sent 3v2 as 3v5 => [3:[1], 1:[3,2,1],2:[4,3,2]]
  this->callback(true);   // sent 3v1 as 3v5 => [1:[3,2,1],2:[4,3,2]]
  this->callback(true);   // sent 1v3 as 1v5 => [1:[2,1],2:[4,3,2]]
  this->callback(true);   // sent 1v2 as 1v5 => [1:[1],2:[4,3,2]]
  this->callback(true);   // sent 1v1 as 1v5 => [2:[4,3,2]]
  this->callback(true);   // sent 2v4 as 2v5 => [2:[3,2]]
  this->callback(true);   // sent 2v3 as 2v5 => [2:[2]]
  this->callback(true);   // sent 2v2 as 2v5 => []
  this->callback(true);   // nothing to send
}

TYPED_TEST(BlobCommsTest, cancelDuringReceivingPartsPushedLayer) {
  this->buildOne();
  int part_id = 0;  // blob_id = 0, , version = 1;
  {
    InSequence dummy;
    this->buildSendMethodExpects(0,  0, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(0,  1, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(0,  2, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(0,  3, part_id, 1, &this->callback, 1);
    EXPECT_CALL(*this->blob_accessor_mock, get_blob(0, 4)).Times(0);
    EXPECT_CALL(*this->blob_accessor_mock, get_blob(0, 5)).Times(0);
  }
  this->comms->push(0, 1);    // 0b0 added to send queue and sent immediately
  this->callback(true);       // 0b1 sent
  this->callback(true);       // 0b2 sent
  this->callback(true);       // 0b3 sent
  this->comms->cancel(0, 1);  // sets 'during_sending' state to false
  this->callback(true);       // clears cancelled version
}
TYPED_TEST(BlobCommsTest, pushLayers) {
  this->buildOne();
  int part_id = 0;  // blob_id = 0, , version = 1;
  {
    InSequence dummy;
    this->buildSendMethodExpects(0,  0, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(0,  1, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(0,  2, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(0,  3, part_id, 1, &this->callback, 1);
    EXPECT_CALL(*this->blob_accessor_mock, get_blob(0, 4)).Times(0);
    EXPECT_CALL(*this->blob_accessor_mock, get_blob(0, 5)).Times(0);
    this->buildSendMethodExpects(1,  0, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(1,  1, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(1,  2, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(1,  3, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(1,  4, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(1,  5, part_id, 1, &this->callback, 1);

    this->buildSendMethodExpects(2,  0, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(2,  1, part_id, 1, &this->callback, 1);

    this->buildSendMethodExpects(3,  0, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(3,  1, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(3,  2, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(3,  3, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(3,  4, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(3,  5, part_id, 1, &this->callback, 1);

    this->buildSendMethodExpects(2,  2, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(2,  3, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(2,  4, part_id, 1, &this->callback, 1);
    this->buildSendMethodExpects(2,  5, part_id, 1, &this->callback, 1);
  }
  this->comms->push(0, 1);    // 0b0 added to send queue and sent immediately
  this->callback(true);       // 0b1 sent
  this->callback(true);       // 0b2 sent
  this->callback(true);       // 0b3 sent
  this->comms->cancel(0, 1);  // sets 'during_sending' state to false
  this->callback(true);       // clears cancelled version
  this->comms->push(1, 1);    // 1b0 sent
  this->callback(true);       // 1b1 sent
  this->callback(true);       // 1b2 sent
  this->callback(true);       // 1b3 sent
  this->callback(true);       // 1b4 sent
  this->callback(true);       // 1b5 sent
  this->comms->push(2, 1);    // just adds to queue bof 'during_sending' state
  this->callback(true);       // 2b0 sent
  this->callback(true);       // 2b1 sent
  this->comms->push(3, 1);    // // adds to queue bof 'during_sending' state
  this->callback(true);       // 3b0 sent
  this->callback(true);       // 3b1 sent
  this->callback(true);       // 3b2 sent
  this->callback(true);       // 3b3 sent
  this->callback(true);       // 3b4 sent
  this->callback(true);       // 3b5 sent

  this->callback(true);       // 2b2 sent
  this->callback(true);       // 2b3 sent
  this->callback(true);       // 2b4 sent
  this->callback(true);       // 2b5 sent
  this->callback(true);       // nothing to send
}

TYPED_TEST(BlobCommsTest, receiveProperBlobUpdate) {
  int layer_id= 0, part_id = 0, blob_id = 0, version = 1;
  this->buildOne();
  BlobUpdate update;
  update.mutable_info()->set_layer_id(layer_id);
  update.mutable_info()->set_blob_id(blob_id);
  update.mutable_info()->set_part(part_id);
  update.mutable_info()->set_version(version);

  this->codec_mock->encode_real(
      &update, &this->blob_accessor_mock->dummy_blob,
      this->settings.what_sent, part_id);

  string str = update.SerializeAsString();
  vector<char> dane(str.begin(), str.end());
  size_t waypoint_id = 0;
//    size_t waypoint_id = waypoint_mock->id();
  vector<int> v(boost::assign::list_of(1).operator vector<int> ());
  Blob<float > blob(v);

  {
    InSequence dumm;
    EXPECT_CALL(*this->waypoint_mock, id());
    EXPECT_CALL(*this->sync_info_mock,
      received_version(waypoint_id, layer_id, blob_id, part_id)).Times
      (AtLeast(1));

    const BlobUpdateInfoEqRefMatcherP4<int, int, int, int> &p4 =
    BlobUpdateInfoEqRef(layer_id, blob_id, part_id, version);

    EXPECT_CALL(*this->keychain_mock, lock(layer_id));
    EXPECT_CALL(*this->codec_mock, decode(p4, _,
                            this->settings.what_received,
                            this->settings.received_incoming_multiplier,
                            this->settings.received_current_multiplier));
    EXPECT_CALL(*this->keychain_mock, unlock(layer_id));

    EXPECT_CALL(*this->sync_info_mock,
      received(waypoint_id, layer_id, blob_id, part_id, version));
  }
  this->comms->received(&dane[0], str.size(), this->waypoint_mock.get());
}

TYPED_TEST(BlobCommsTest, receiveWrongBlobUpdate) {
  this->buildOne();
  vector<int> v(boost::assign::list_of(1).operator vector<int> ());
  Blob<float > blob(v);

  EXPECT_CALL(*this->waypoint_mock, id()).Times(1);
  EXPECT_CALL(*this->sync_info_mock, received_version(_, _, _, _)).Times(0);

  EXPECT_CALL(*this->sync_info_mock, received(_, _, _, _, _)).Times(0);

  EXPECT_CALL(*this->blob_accessor_mock, get_blob(_, _)).Times(0);
  EXPECT_CALL(*this->keychain_mock, lock(_)).Times(0);
  EXPECT_CALL(*this->codec_mock, decode(_, _, _, _, _)).Times(0);
  EXPECT_CALL(*this->keychain_mock, unlock(_)).Times(0);

  this->comms->received(&vector<char>(boost::assign::list_of(1).operator
      vector<char>
      ())[0], 3, this->waypoint_mock.get());
}

TYPED_TEST(BlobCommsTest, receiveBlobUpdateWithoutInfo) {
  int part_id = 0;
  this->buildOne();
  BlobUpdate update;

  EXPECT_CALL(*this->waypoint_mock, id()).Times(1);
  EXPECT_CALL(*this->sync_info_mock, received_version(0, 0, 0, 0)).Times(1);

  EXPECT_CALL(*this->sync_info_mock, received(_, _, _, _, _)).Times(0);

  EXPECT_CALL(*this->blob_accessor_mock, get_blob(_, _)).Times(0);
  EXPECT_CALL(*this->keychain_mock, lock(_)).Times(0);
  EXPECT_CALL(*this->codec_mock, decode(_, _, _, _, _)).Times(0);
  EXPECT_CALL(*this->keychain_mock, unlock(_)).Times(0);

  this->codec_mock->encode_real(
      &update, &this->blob_accessor_mock->dummy_blob,
      this->settings.what_sent, part_id);

  string str = update.SerializeAsString();
  vector<char> dane(str.begin(), str.end());

  vector<int> v(boost::assign::list_of(1).operator vector<int> ());
  Blob<float > blob(v);

  this->comms->received(&dane[0], str.size(), this->waypoint_mock.get());
}

TYPED_TEST(BlobCommsTest, receiveBlobUpdateWithIters) {
  int iters_count = 2;
  size_t remote_id = 0;
  this->buildOne();
  BlobUpdate update;

  this->comms->register_iter_size_handler(this->iter_size_handler_mock1.get());
  this->comms->register_iter_size_handler(this->iter_size_handler_mock2.get());
  update.set_iters(iters_count);
  update.clear_info();

  string str = update.SerializeAsString();
  vector<char> dane(str.begin(), str.end());
  vector<int> v(boost::assign::list_of(1).operator vector<int> ());
  Blob<float > blob(v);

  EXPECT_CALL(*this->waypoint_mock, id()).Times(1);
  EXPECT_CALL(*this->iter_size_handler_mock1,
              received_iter_size(remote_id, iters_count)).Times(1);
  EXPECT_CALL(*this->iter_size_handler_mock2,
              received_iter_size(remote_id, iters_count)).Times(1);
//      should return from method
//      the following not to be called
  EXPECT_CALL(*this->sync_info_mock, received_version(_, _, _, _)).Times(0);
  EXPECT_CALL(*this->sync_info_mock, received(_, _, _, _, _)).Times(0);

  EXPECT_CALL(*this->blob_accessor_mock, get_blob(_, _)).Times(0);
  EXPECT_CALL(*this->keychain_mock, lock(_)).Times(0);
  EXPECT_CALL(*this->codec_mock, decode(_, _, _, _, _)).Times(0);
  EXPECT_CALL(*this->keychain_mock, unlock(_)).Times(0);

  this->comms->received(&dane[0], str.size(), this->waypoint_mock.get());
}
TYPED_TEST(BlobCommsTest, receiveBlobUpdateWithNoIters) {
  this->buildOne();
  BlobUpdate update;
  update.clear_iters();
  update.clear_info();

  string str = update.SerializeAsString();
  vector<char> dane(str.begin(), str.end());
  vector<int> v(boost::assign::list_of(1).operator vector<int> ());
  Blob<float> blob(v);

  EXPECT_CALL(*this->waypoint_mock, id()).Times(2);
//      should return from method
//      the following not to be called
  EXPECT_CALL(*this->iter_size_handler_mock1, received_iter_size(_, _))
               .Times(0);
  EXPECT_CALL(*this->sync_info_mock, received_version(_, _, _, _)).Times(0);
  EXPECT_CALL(*this->sync_info_mock, received(_, _, _, _, _)).Times(0);

  EXPECT_CALL(*this->blob_accessor_mock, get_blob(_, _)).Times(0);
  EXPECT_CALL(*this->keychain_mock, lock(_)).Times(0);
  EXPECT_CALL(*this->codec_mock, decode(_, _, _, _, _)).Times(0);
  EXPECT_CALL(*this->keychain_mock, unlock(_)).Times(0);

  this->comms->received(&dane[0], str.size(), this->waypoint_mock.get());
  this->comms->received(&dane[0], str.size(), this->waypoint_mock.get());
}
}  // namespace
}  // namespace caffe
