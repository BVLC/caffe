#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/serialization/bitfield.hpp"
#include "caffe/blob.hpp"
#include "caffe/multinode/BlobComms.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/internode/configuration.hpp"

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

struct BlobCommsTest : TerminatedHandler, public Test, InternalThread {
  shared_ptr<internode::Daemon> comm;
  shared_ptr<BlobCodec<float> > codec;
  shared_ptr<internode::Waypoint> waypoint;

  shared_ptr<BlobConstInfoMock> const_info;
  shared_ptr<SyncedMock> sync_info;
  shared_ptr<BlobKeyChain<float> > keychain;
  shared_ptr<BlobComms<float> > comms;
  shared_ptr<MultinodeParam> param;

  std::vector<LayerState> layers;
  LayerState init;

  boost::mutex mtx;
  bool terminated_;
  
  string address;
  int num_of_threads;
  
    virtual void SetUp() {
    const_info.reset(new BlobConstInfoMock());
    sync_info.reset(new StrictMock<SyncedMock>());
  }

  virtual void TearDown() {
    const_info.reset();
    sync_info.reset();
  }
  
  BlobCommsTest()
    : comm(internode::create_communication_daemon())
    , param(MultinodeParameter::default_instance())
    , codec(BlobCodec<float>::create_codec(param, true))
    , waypoint(internode::configure_client(comm, address, codec->packet_size()))
    , keychain(BlobKeyChain<float>::create_empty(const_info->layers()))
    , comms(
        BlobComms<float>::create(
          solver, const_info, sync_info, waypoint, codec, keychain,
          typename BlobComms<float>::Settings(
            BlobEncoding::GRADS, BlobEncoding::PARAMS, 1.0, 0.0),
          num_of_threads))
    , layers(const_info->layers())
    , terminated_(false) {
    init.move_to(LayerState::updating);
    waypoint->register_receive_handler(comms.get());
    comms->send_iter_size(param->iter_size());

    internode::create_timer(
      comm,
      500000,
      boost::bind(&SynchronousParamSyncingImpl::tick, this),
      true);
  }
  
TEST_F(BlobCommsTest, LOL_test1)
{

}
        
}  // namespace
}  // namespace caffe
