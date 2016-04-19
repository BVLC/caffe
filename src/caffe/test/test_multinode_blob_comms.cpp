#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/serialization/bitfield.hpp"
#include "caffe/blob.hpp"
#include "caffe/multinode/BlobComms.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/internode/configuration.hpp"
#include "caffe/solver_factory.hpp"

namespace caffe {
namespace {

using ::testing::_;
using ::testing::Return;
using ::testing::Test;
using ::testing::StrictMock;
using ::testing::Mock;

struct BlobConstInfoMock : BlobConstInfo {
  MOCK_CONST_METHOD2(parts, uint32_t(int a, int b));
  MOCK_CONST_METHOD1(parts, uint32_t(int a));
  MOCK_CONST_METHOD0(parts, uint32_t());
  MOCK_CONST_METHOD1(blobs, uint32_t(int a));
  MOCK_CONST_METHOD0(layers, uint32_t());
  MOCK_CONST_METHOD1(needs_syncing, bool(int a));
};

struct SyncedMock : public BlobSyncInfo::Handler {
  MOCK_METHOD4(synced, void(int a, int b, int c, uint32_t d));
  MOCK_METHOD2(synced, void(int a, uint32_t b));
  MOCK_METHOD1(synced, void(uint32_t a));
};

struct BlobCommsTest : public Test {
  shared_ptr<internode::Daemon> comm;
  shared_ptr<BlobCodec<float> > codec;
  shared_ptr<internode::Waypoint> waypoint;

  shared_ptr<BlobConstInfoMock> const_info_mock;
  shared_ptr<SyncedMock> sync_mock;
  shared_ptr<BlobSyncInfo> sync_info;
  shared_ptr<BlobKeyChain<float> > keychain;
  shared_ptr<BlobComms<float> > comms;
  shared_ptr<Solver<float> > solver;
  
  string address;
  int num_of_threads;
  
    virtual void SetUp() {
    const_info_mock.reset(new BlobConstInfoMock());
    sync_mock.reset(new StrictMock<SyncedMock>());
    sync_info = BlobInfoFactory<float>::create_sync_info(const_info_mock);
    //if (register_handler)
      sync_info->register_synced_handler(sync_mock.get());
    EXPECT_CALL(*const_info_mock, layers()).WillRepeatedly(Return(1));
  }

  virtual void TearDown() {
    sync_info.reset();
    sync_mock.reset();
    const_info_mock.reset();
  }
  
  void buildOne(){
    LOG(INFO) << "1";
    SolverParameter solverparam = SolverParameter::default_instance();
    solverparam.set_train_net("1");
    solver.reset(SolverRegistry<float>::CreateSolver(solverparam));
    LOG(INFO) << "2";
    comm = internode::create_communication_daemon();
    codec = BlobCodec<float>::create_codec(MultinodeParameter::default_instance(), true);
    waypoint = internode::configure_client(comm, address, codec->packet_size());
    keychain = BlobKeyChain<float>::create_empty(const_info_mock->layers());
    comms = BlobComms<float>::create(
            solver, const_info_mock, sync_info, waypoint, codec, keychain,
            typename BlobComms<float>::Settings(
              BlobEncoding::GRADS, BlobEncoding::PARAMS, 1.0, 0.0),
            num_of_threads);

      waypoint->register_receive_handler(comms.get());
      comms->send_iter_size(solver->param().iter_size());      
  } 
};

TEST_F(BlobCommsTest, LOL_test1)
{
    buildOne();
    EXPECT_EQ(0, comms->currently_sending_version());
}
        
}  // namespace
}  // namespace caffe
