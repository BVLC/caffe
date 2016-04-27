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
// #include "caffe/multinode/SendCallback.hpp"

namespace caffe {
namespace {

using ::testing::_;
using ::testing::Return;
using ::testing::Test;
using ::testing::StrictMock;
using ::testing::Mock;
using ::testing::SaveArg;
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

    const_info_mock.reset(new StrictMock<BlobConstInfoMock>());
    prepare_const_mock(list_of<vector<int> >(list_of<int>(1)));

    blob_accessor_mock.reset(new StrictMock<BlobAccessorMock<float> >());
    sync_info.reset(new StrictMock<BlobSyncInfoMock>());
  }

  BlobCommsTest()
      : num_of_threads(1) {}

  virtual void TearDown() {
    sync_info.reset();
    blob_accessor_mock.reset();
    const_info_mock.reset();
    waypoint_mock.reset();
  }

  void buildOne() {
    codec = BlobCodec<float>::create_codec(
            MultinodeParameter::default_instance(), true);
    keychain = BlobKeyChain<float>::create_empty(const_info_mock->layers());
    comms = BlobComms<float>::create(blob_accessor_mock,
            const_info_mock, sync_info, waypoint_mock, codec, keychain,
            typename BlobComms<float>::Settings(
              BlobEncoding::GRADS, BlobEncoding::PARAMS, 1.0, 0.0),
            num_of_threads);
  }
};

TEST_F(BlobCommsTest, CurrentlySendingVersionSizeCheck) {
    buildOne();
    EXPECT_EQ(0, comms->currently_sending_version());
    EXPECT_EQ(0, comms->currently_sending_version(0));
    EXPECT_DEATH(comms->currently_sending_version(1), "");
}

MATCHER_P(BufferEq, str, "") {
  return std::equal(str.c_str(), str.c_str() + str.size(), arg);
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

}  // namespace
}  // namespace caffe
