#include <boost/assign.hpp>
#include <boost/make_shared.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/internode/communication.hpp"
#include "caffe/internode/configuration.hpp"
#include "caffe/internode/tree_cluster.hpp"
#include "caffe/multinode/BlobComms.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/serialization/bitfield.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/solver_factory.hpp"

namespace caffe {
namespace {

using ::testing::_;
using ::testing::Return;
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
  MOCK_METHOD1(unlock,void (int layer_id));
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
      ON_CALL(*this, encode(_,_,_,_))
        .WillByDefault(Invoke(real_.get(), &BlobCodec<Dtype>::encode));
      ON_CALL(*this, decode(_,_,_,_,_))
        .WillByDefault(Invoke(real_.get(), &BlobCodec<Dtype>::decode));
      ON_CALL(*this, max_elements_per_part())
        .WillByDefault(Invoke(real_.get(), &BlobCodec<Dtype>::max_elements_per_part));
      ON_CALL(*this, packet_size())
        .WillByDefault(Invoke(real_.get(), &BlobCodec<Dtype>::packet_size));
     };
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
class BlobAccessorMock : public BlobAccessor<Dtype> {
  vector<int> v;
  Blob<Dtype> blob;
 public:
  BlobAccessorMock()
  : v(boost::assign::list_of(1)(1)(1)(1)(1)(1)(1)(1).operator vector<int> ())
  , blob(v) {}

  virtual Blob<Dtype>* get_blob(int layer, int blob_id) {
    return &blob;
  }
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

struct BlobCommsTest : public Test {
//  shared_ptr<BlobCodec<float> > codec;
  shared_ptr<BlobCodecMock<float> > codec_mock;
  shared_ptr<WaypointMock> waypoint_mock;

  shared_ptr<BlobAccessorMock<float> > blob_accessor_mock;
  shared_ptr<BlobConstInfoMock> const_info_mock;
  shared_ptr<BlobSyncInfoMock> sync_info;
  shared_ptr<BlobKeyChain<float> > keychain;
  shared_ptr<BlobKeyChainMock<float> > keychain_mock;
  shared_ptr<BlobComms<float> > comms;
  BlobComms<float>::Settings settings;
  int num_of_threads;

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
    
    prepare_const_mock(
        list_of<vector<int> >
            (list_of<int>(1)(1)(1)(1)(1)(1))
            (list_of<int>(2)(2)(2)(2)(2)(2))
            (list_of<int>(3)(3)(3)(3)(3)(3))
            (list_of<int>(4)(4)(4)(4)(4)(4))
    );
    
    keychain_mock.reset(new StrictMock<BlobKeyChainMock<float> >() );

    blob_accessor_mock.reset(new StrictMock<BlobAccessorMock<float> >());
    sync_info.reset(new StrictMock<BlobSyncInfoMock>());
  }

  BlobCommsTest()
      : settings(BlobComms<float>::Settings(
              BlobEncoding::GRADS, BlobEncoding::PARAMS, 1.0, 0.0))
        , num_of_threads(1){}

  virtual void TearDown() {
    codec_mock.reset();
    sync_info.reset();
    blob_accessor_mock.reset();
    keychain_mock.reset();
    const_info_mock.reset();
    waypoint_mock.reset();
  }

    void buildSendMethodExpectations(
          int layer_id, 
          int blob_id, 
          int part_id, 
          uint32_t version,
          Waypoint::SentCallback* callback,
          int times){
        BlobUpdate update;
        update.mutable_info()->set_layer_id(layer_id);
        update.mutable_info()->set_blob_id(blob_id);
        update.mutable_info()->set_part(part_id);
        update.mutable_info()->set_version(version);  
        buildSendMethodExpectations(layer_id, blob_id, part_id, 
            version, callback, update, times);
    };
  void buildSendMethodExpectations(
          int layer_id, 
          int blob_id, 
          int part_id, 
          uint32_t version,
          Waypoint::SentCallback* callback,
          BlobUpdate& update,
          int times){
  
    codec_mock->encode(
        &update, blob_accessor_mock->get_blob(layer_id, blob_id), 
        settings.what_sent, update.info().part());     
    
    string str = update.SerializeAsString();

    
    InSequence dummy;
    for (int i =0; i< times;++i) {
    EXPECT_CALL(*keychain_mock, 
                lock(layer_id));
//        .Times(times);    
    EXPECT_CALL(*codec_mock, 
                encode(
                    BlobUpdateInfoEq(layer_id, blob_id, part_id, version),
                    blob_accessor_mock->get_blob(layer_id, blob_id),
                    BlobEncoding::GRADS, part_id));
//        .Times(times);        
    EXPECT_CALL(*keychain_mock, 
                unlock(layer_id));
//        .Times(times);  
//    if(times <= 0)
//        EXPECT_CALL(*waypoint_mock, 
//                    async_send(BufferEq(str), update.ByteSize(), _))
//            .Times(0);          
//    else
        EXPECT_CALL(*waypoint_mock, 
                    async_send(BufferEq(str), update.ByteSize(), _))
//            .Times(times)
            .WillOnce(SaveArg<2>(callback));          
    }
  }
  
  void buildOne() {
//    codec = BlobCodec<float>::create_codec(
//            MultinodeParameter::default_instance(), true);
    keychain = BlobKeyChain<float>::create_empty(const_info_mock->layers());
//    settings = BlobComms<float>::Settings(
//              BlobEncoding::GRADS, BlobEncoding::PARAMS, 1.0, 0.0);
    comms = BlobComms<float>::create(blob_accessor_mock,
            const_info_mock, sync_info, waypoint_mock, codec_mock, keychain_mock,
           settings, num_of_threads);
  }
};

TEST_F(BlobCommsTest, CurrentlySendingVersionSizeCheck) {
    buildOne();
    EXPECT_EQ(0, comms->currently_sending_version());
    EXPECT_EQ(0, comms->currently_sending_version(0));
    EXPECT_DEATH(comms->currently_sending_version(1), "");
}


void SendIterSize(
        shared_ptr<BlobComms<float> > comms,
        shared_ptr<WaypointMock> waypoint_mock,
        int size) {
  BlobUpdate update;
  update.set_iters(size);
  string str = update.SerializeAsString();
  Waypoint::SentCallback callback;
  EXPECT_CALL(*waypoint_mock, async_send(BufferEq(str), update.ByteSize(), _))
      .WillOnce(SaveArg<2>(&callback));
  comms->send_iter_size(size);
  callback(true);
}

TEST_F(BlobCommsTest, SendIterSize) {
  buildOne();

  SendIterSize(comms, waypoint_mock, 10);
  SendIterSize(comms, waypoint_mock, -1);
  SendIterSize(comms, waypoint_mock, 0);
  SendIterSize(comms, waypoint_mock, 101);
  SendIterSize(comms, waypoint_mock, 1);
}
TEST_F(BlobCommsTest, pushOneWithCancelledVersion) {
  buildOne();
  int layer_id = 0, blob_id = 0, part_id = 0, version = 1;
  int times = 0;

  buildSendMethodExpectations(layer_id, blob_id, part_id, version, NULL, times);
  comms->cancel(layer_id, version);
  comms->push(layer_id, blob_id, part_id, version);
}
TEST_F(BlobCommsTest, pushOne) {
  buildOne();
  int layer_id = 0, blob_id = 0, part_id = 0, version = 1;
  int times = 1;
  Waypoint::SentCallback callback;
  
  buildSendMethodExpectations(layer_id, blob_id, part_id, 
            version, &callback, times);    
  comms->push(layer_id, blob_id, part_id, version);
  callback(true);
}
TEST_F(BlobCommsTest, pushAnotherTwoDuringSending) {
  buildOne();
  int layer_id = 0, blob_id = 0, part_id = 0, version = 1;
  int times = 1;
  Waypoint::SentCallback callback;
  
  buildSendMethodExpectations(layer_id, blob_id, part_id, 
            version, &callback, times);    
  comms->push(layer_id, blob_id, part_id, version);
  comms->push(layer_id, blob_id, part_id, version);
  comms->push(layer_id, blob_id, part_id, version);

}
TEST_F(BlobCommsTest, push3OneByOne) {
  buildOne();
  int layer_id = 0, blob_id = 0, part_id = 0, version = 1;
  int times = 3;
  Waypoint::SentCallback callback;
  
  buildSendMethodExpectations(layer_id, blob_id, part_id, 
            version, &callback, times);    
  comms->push(layer_id, blob_id, part_id, version);
  //simulate Waypoint async_send => implicit call BlobComms::sent()
  //clears during_sending BlobComms state
  callback(true); 
  comms->push(layer_id, blob_id, part_id, version);
  callback(true);
  comms->push(layer_id, blob_id, part_id, version);
  callback(true);

}
TEST_F(BlobCommsTest, cancelOneWhenInQueueDuringSending3Queue) {
  buildOne();
  int layer_id = 0, blob_id = 0, part_id = 0, version = 1;
  int times = 2;
  Waypoint::SentCallback callback;
  
  buildSendMethodExpectations(layer_id, blob_id, part_id, 
            version, &callback, times);    
  comms->push(layer_id, blob_id, part_id, version);
  //simulate Waypoint async_send => implicit call BlobComms::sent()
  //clears during_sending BlobComms state
  callback(true); 
  comms->push(layer_id, blob_id, part_id, version);
  callback(true); 
  comms->cancel(layer_id, version);
  comms->push(layer_id, blob_id, part_id, version);
  callback(true); 
}
TEST_F(BlobCommsTest, cancelLayer1WhenInQueue) {
  buildOne();
  int blob_id = 0, part_id = 0, version = 1;
//  int times = 2;
  Waypoint::SentCallback callback;

//  buildSendMethodExpectations(1, blob_id, part_id, 
//            1, &callback, 0);    
//  buildSendMethodExpectations(2, blob_id, part_id, 
//            2, &callback, 2);    
  
//  comms->cancel(1, version);
  comms->push(6, 4, 9, 6);
//  callback(true); 
//  comms->push(1, blob_id, part_id, 1);
  comms->push(4, blob_id, part_id, 1);
  comms->push(2, blob_id, part_id, 2);
//simulate Waypoint async_send => implicit call BlobComms::sent()
  //clears during_sending BlobComms state
//  callback(true); 
//  callback(true); 
//  callback(true); 
//  callback(true); 
}
TEST_F(BlobCommsTest, checkPriorityQueue) {
  buildOne();
  int blob_id = 0, part_id = 0, version = 1;
//  int times = 2;
  Waypoint::SentCallback callback;

//  buildSendMethodExpectations(1, blob_id, part_id, 
//            1, &callback, 0);    
//  buildSendMethodExpectations(2, blob_id, part_id, 
//            2, &callback, 2);    
  
//  comms->push(2, blob_id, part_id, 2);
//  comms->push(2, blob_id, part_id, 2);
//  comms->push(2, blob_id, part_id, 1);
//  comms->push(2, blob_id, part_id, 1);
//  comms->push(2, blob_id, part_id, 2);
//  comms->push(2, blob_id, part_id, 2);
//simulate Waypoint async_send => implicit call BlobComms::sent()
  //clears during_sending BlobComms state
//  callback(true); 
//  callback(true); 
//  callback(true); 
//  callback(true); 
}
}  // namespace
}  // namespace caffe
