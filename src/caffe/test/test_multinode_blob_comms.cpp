#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <boost/assign.hpp>
#include <boost/make_shared.hpp>
#include "caffe/blob.hpp"
#include "caffe/internode/configuration.hpp"
#include "caffe/multinode/BlobComms.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/serialization/bitfield.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/solver_factory.hpp"
#include "caffe/internode/tree_cluster.hpp"
#include "caffe/internode/communication.hpp"

namespace caffe {
namespace {

using ::testing::_;
using ::testing::Return;
using ::testing::Test;
using ::testing::StrictMock;
using ::testing::Mock;
using ::boost::assign::list_of;

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
class BlobAccessorMock : public BlobAccessor<Dtype> {
  vector<int> v;
  Blob<Dtype> blob;
 public:
  BlobAccessorMock() 
  : v(boost::assign::list_of(1)(1)(1)(1).operator vector<int> ())
  , blob(v) {}
    
  virtual Blob<Dtype>* get_blob(int layer, int blob_id) {
    return &blob;
  }
};

struct BlobCommsTest : public Test {
  shared_ptr<BlobCodec<float> > codec;
  shared_ptr<WaypointMock> waypoint_mock;

  shared_ptr<BlobAccessorMock<float> > blob_accessor_mock;
  shared_ptr<BlobConstInfoMock> const_info_mock;
  shared_ptr<BlobSyncInfoMock> sync_info;
  shared_ptr<BlobKeyChain<float> > keychain;
  shared_ptr<BlobComms<float> > comms;

  int num_of_threads;
  
  void prepare_const_mock(
      vector<vector<int> > vparts, bool register_handler = true) {
    EXPECT_CALL(*const_info_mock, layers()).WillRepeatedly(Return(vparts.size()));
    for (int i = 0; i < vparts.size(); ++i) {
      EXPECT_CALL(*const_info_mock, blobs(i)).WillRepeatedly(Return(vparts.at(i).size()));
      int totalpartsinlayer = 0;
      for (int j = 0; j < vparts.at(i).size(); ++j) {
        EXPECT_CALL(*const_info_mock, parts(i, j))
          .WillRepeatedly(Return(vparts.at(i).at(j)));
        totalpartsinlayer+=vparts.at(i).at(j);
      }
      EXPECT_CALL(*const_info_mock, parts(i)).WillRepeatedly(Return(totalpartsinlayer));
    }
    EXPECT_CALL(*const_info_mock, needs_syncing(_)).WillRepeatedly(Return(true));
  }

  virtual void SetUp() {
    const_info_mock.reset(new BlobConstInfoMock());
    prepare_const_mock(list_of<vector<int> >(list_of<int>(1)));
    
    blob_accessor_mock.reset(new BlobAccessorMock<float>());
    sync_info.reset(new BlobSyncInfoMock());
  }
  
  BlobCommsTest()
      : num_of_threads(1) {
 }

  virtual void TearDown() {
    sync_info.reset();
    blob_accessor_mock.reset();
    const_info_mock.reset();
  }

  void buildOne() {
    //comm = internode::create_communication_daemon();
    codec = BlobCodec<float>::create_codec(
            MultinodeParameter::default_instance(), true);
    waypoint_mock.reset(new WaypointMock());
    keychain = BlobKeyChain<float>::create_empty(const_info_mock->layers());
    comms = BlobComms<float>::create(blob_accessor_mock,
            const_info_mock, sync_info, waypoint_mock, codec, keychain,
            typename BlobComms<float>::Settings(
              BlobEncoding::GRADS, BlobEncoding::PARAMS, 1.0, 0.0),
            num_of_threads);

    //waypoint_mock->register_receive_handler(comms.get());
    //comms->send_iter_size();
  }
};

TEST_F(BlobCommsTest, CurrentlySendingVersionSizeCheck) {
    buildOne();
    EXPECT_EQ(0, comms->currently_sending_version());
    EXPECT_EQ(0, comms->currently_sending_version(0));
    EXPECT_DEATH(comms->currently_sending_version(1), "");
}

TEST_F(BlobCommsTest, SendIterSize) {
    buildOne();
    comms->send_iter_size(1);
//    EXPECT_CALL(waypoint_mock, async_send(
//            A<const char*>(),
//            A<size_t>(),
//            A<Waypoint::SentCallback>()))
//        .WithArg<2>().WillRepeatedly(InvokeCallback);
    EXPECT_CALL(waypoint_mock, async_send(_,_,_))
        .WillRepeatedly(testing::InvokeArgument<2>(true));
    EXPECT_EQ(1, comms->currently_sending_version());
   comms->send_iter_size(2);
    EXPECT_EQ(2, comms->currently_sending_version());
   comms->send_iter_size(3);
    EXPECT_EQ(3, comms->currently_sending_version());
}

}  // namespace
}  // namespace caffe
