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
#include "caffe/multinode/BlobInfo.hpp"

namespace caffe {
namespace {

using ::testing::_;
using ::testing::Return;
using ::testing::Test;
using ::testing::StrictMock;
using ::testing::Mock;
using ::testing::InSequence;
using boost::assign::list_of;

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

struct SyncBlobInfoTest : public Test {
  shared_ptr<BlobConstInfoMock> mock;
  shared_ptr<SyncedMock> sync_mock;

  shared_ptr<BlobSyncInfo> prepare_const_mock(
      vector<vector<int> > vparts, bool register_handler = true) {
    EXPECT_CALL(*mock, layers()).WillRepeatedly(Return(vparts.size()));
    for (int i = 0; i < vparts.size(); ++i) {
      EXPECT_CALL(*mock, blobs(i)).WillRepeatedly(Return(vparts.at(i).size()));
      int totalpartsinlayer = 0;
      for (int j = 0; j < vparts.at(i).size(); ++j) {
        EXPECT_CALL(*mock, parts(i, j))
          .WillRepeatedly(Return(vparts.at(i).at(j)));
        totalpartsinlayer+=vparts.at(i).at(j);
      }
      EXPECT_CALL(*mock, parts(i)).WillRepeatedly(Return(totalpartsinlayer));
    }
    EXPECT_CALL(*mock, needs_syncing(_)).WillRepeatedly(Return(true));
    shared_ptr<BlobSyncInfo> sync_info =
      BlobInfoFactory<float>::create_sync_info(mock);
    if (register_handler)
      sync_info->register_synced_handler(sync_mock.get());
    return sync_info;
  }

  virtual void SetUp() {
    mock.reset(new BlobConstInfoMock());
    sync_mock.reset(new StrictMock<SyncedMock>());
  }

  virtual void TearDown() {
    mock.reset();
    sync_mock.reset();
  }

  ~SyncBlobInfoTest() {
    mock.reset();
    sync_mock.reset();
  }
};

TEST_F(SyncBlobInfoTest, ReceivedVersionSingleRemote) {
  shared_ptr<BlobSyncInfo> sync = prepare_const_mock(
          list_of<vector<int> >(list_of<int>(1)));
  ASSERT_EQ(0, sync->received_version(0, 0, 0, 0));
  sync->add_remote(0);
  ASSERT_EQ(0, sync->received_version(0, 0, 0, 0));
  {
    InSequence seq;
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 1));
    EXPECT_CALL(*sync_mock, synced(1));
  }
  sync->received(0, 0, 0, 0, 1);
  Mock::VerifyAndClearExpectations(&sync_mock);
  ASSERT_EQ(1, sync->received_version(0, 0, 0, 0));
  {
    InSequence seq;
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 2));
    EXPECT_CALL(*sync_mock, synced(0, 2));
    EXPECT_CALL(*sync_mock, synced(2));
  }
  sync->received(0, 0, 0, 0, 2);
  Mock::VerifyAndClearExpectations(&sync_mock);
  ASSERT_EQ(2, sync->received_version(0, 0, 0, 0));
  {
    InSequence seq;
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 2));
    EXPECT_CALL(*sync_mock, synced(0, 2));
    EXPECT_CALL(*sync_mock, synced(2));
  }
  sync->remove_remote(0);
  Mock::VerifyAndClearExpectations(&sync_mock);
  ASSERT_EQ(1, sync->received_version(0, 0, 0, 0));
}

TEST_F(SyncBlobInfoTest, ReceivedVersionMultipleRemote) {
  shared_ptr<BlobSyncInfo> sync = prepare_const_mock(
          list_of<vector<int> >(list_of<int>(1)));
  for (int i = 0; i < 5; i++)
    sync->add_remote(i);
  {
    InSequence seq;
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 1));
    EXPECT_CALL(*sync_mock, synced(1));
  }
  for (int i = 0; i < 5; i++)
    sync->received(i, 0, 0, 0, 1);
  Mock::VerifyAndClearExpectations(&sync_mock);
  ASSERT_EQ(1, sync->received_version(0, 0, 0, 0));
  for (int i = 0; i < 3; i++)
    sync->received(i, 0, 0, 0, 2);
  sync->remove_remote(4);
  ASSERT_EQ(2, sync->received_version(0, 0, 0, 0));
  sync->remove_remote(0);
  ASSERT_EQ(1, sync->received_version(0, 0, 0, 0));
  {
    InSequence seq;
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 2));
    EXPECT_CALL(*sync_mock, synced(0, 2));
    EXPECT_CALL(*sync_mock, synced(2));
  }
  sync->remove_remote(3);
  Mock::VerifyAndClearExpectations(&sync_mock);
  ASSERT_EQ(1, sync->received_version(0, 0, 0, 0));
  ASSERT_EQ(2, sync->received_version(1, 0, 0, 0));
  ASSERT_EQ(2, sync->received_version(2, 0, 0, 0));
  ASSERT_EQ(1, sync->received_version(3, 0, 0, 0));
  ASSERT_EQ(1, sync->received_version(4, 0, 0, 0));
  {
    InSequence seq;
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 2));
    EXPECT_CALL(*sync_mock, synced(0, 2));
    EXPECT_CALL(*sync_mock, synced(2));
  }
  sync->remove_remote(2);
  Mock::VerifyAndClearExpectations(&sync_mock);
  {
    InSequence seq;
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 2));
    EXPECT_CALL(*sync_mock, synced(0, 2));
    EXPECT_CALL(*sync_mock, synced(2));
  }
  sync->remove_remote(1);
  Mock::VerifyAndClearExpectations(&sync_mock);
}

TEST_F(SyncBlobInfoTest, SingleBlobSinglePartSingleRemoteSyncing) {
  shared_ptr<BlobSyncInfo> sync = prepare_const_mock(
          list_of<vector<int> >(list_of<int>(1)));
  sync->add_remote(99);
  {
    InSequence in_seq;
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 1));
    EXPECT_CALL(*sync_mock, synced(1));
  }
  sync->received(99, 0, 0, 0, 1);
}

TEST_F(SyncBlobInfoTest, SingleBlobSinglePartUnregisteredRemoteSyncing) {
  shared_ptr<BlobSyncInfo> sync = prepare_const_mock(
          list_of<vector<int> >(list_of<int>(1)));
  {
    InSequence in_seq;
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 1));
    EXPECT_CALL(*sync_mock, synced(1));
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 1));
    EXPECT_CALL(*sync_mock, synced(1));
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 2));
    EXPECT_CALL(*sync_mock, synced(0, 2));
    EXPECT_CALL(*sync_mock, synced(2));
  }
  sync->add_remote(99);
  sync->received(99, 0, 0, 0, 1);
  sync->received(00, 0, 0, 0, 1);
  sync->received(99, 0, 0, 0, 2);
  sync->received(23, 0, 0, 0, 2);
  sync->received(00, 0, 0, 0, 2);
}

TEST_F(SyncBlobInfoTest, SingleBlobSinglePartTripleRemoteSyncing) {
  shared_ptr<BlobSyncInfo> sync = prepare_const_mock(
          list_of<vector<int> >(list_of<int>(1)));
  {
    InSequence in_seq;
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 2));
    EXPECT_CALL(*sync_mock, synced(0, 2));
    EXPECT_CALL(*sync_mock, synced(2));
  }
  sync->add_remote(-4);
  sync->add_remote(999);
  sync->add_remote(7);
  sync->received(999, 0, 0, 0, 1);
  sync->received(-4,  0, 0, 0, 1);
  sync->received(-4,  0, 0, 0, 1);
  sync->received(999, 0, 0, 0, 2);
  sync->received(7,   0, 0, 0, 1);
  sync->received(-4,  0, 0, 0, 1);
  sync->received(7,   0, 0, 0, 2);
  sync->received(-4,  0, 0, 0, 2);
}

TEST_F(SyncBlobInfoTest, SingleBlobTriplePartSingleRemoteSyncing) {
  shared_ptr<BlobSyncInfo> sync = prepare_const_mock(
          list_of<vector<int> >(list_of<int>(3)));
  {
    InSequence in_seq;
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 0, 1, 1));
    EXPECT_CALL(*sync_mock, synced(0, 0, 2, 1));
    EXPECT_CALL(*sync_mock, synced(0, 1));
    EXPECT_CALL(*sync_mock, synced(1));
  }
  sync->add_remote(0);
  sync->received(0, 0, 0, 0, 1);
  sync->received(0, 0, 0, 1, 1);
  sync->received(0, 0, 0, 2, 1);
}

TEST_F(SyncBlobInfoTest, SingleBlobTriplePartTripleRemoteSyncing) {
  shared_ptr<BlobSyncInfo> sync = prepare_const_mock(
          list_of<vector<int> >(list_of<int>(3)));
  {
    InSequence in_seq;
    EXPECT_CALL(*sync_mock, synced(0, 0, 1, 1));
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 0, 2, 1));
    EXPECT_CALL(*sync_mock, synced(0, 1));
    EXPECT_CALL(*sync_mock, synced(1));
  }
  sync->add_remote(0);
  sync->add_remote(1);
  sync->add_remote(2);
  sync->received(0, 0, 0, 0, 1);
  sync->received(1, 0, 0, 1, 1);
  sync->received(0, 0, 0, 2, 1);
  sync->received(2, 0, 0, 1, 1);
  sync->received(2, 0, 0, 0, 1);
  sync->received(2, 0, 0, 2, 1);
  sync->received(0, 0, 0, 1, 1);
  sync->received(1, 0, 0, 0, 1);
  sync->received(1, 0, 0, 2, 1);
}

TEST_F(SyncBlobInfoTest, TripleBlobSinglePartSingleRemoteSyncing) {
  shared_ptr<BlobSyncInfo> sync = prepare_const_mock(
          list_of<vector<int> >(list_of<int>(1)(1)(1)));
  {
    InSequence in_seq;
    EXPECT_CALL(*sync_mock, synced(0, 1, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 2, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 1));
    EXPECT_CALL(*sync_mock, synced(1));
  }
  sync->add_remote(0);
  sync->received(0, 0, 1, 0, 1);
  sync->received(0, 0, 0, 0, 1);
  sync->received(0, 0, 2, 0, 1);
}
TEST_F(SyncBlobInfoTest, TripleBlobSinglePartTripleRemoteSyncing) {
  shared_ptr<BlobSyncInfo> sync = prepare_const_mock(
          list_of<vector<int> >(list_of<int>(1)(1)(1)));
  {
    InSequence in_seq;
    EXPECT_CALL(*sync_mock, synced(0, 1, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 2, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 1));
    EXPECT_CALL(*sync_mock, synced(1));
  }
  sync->add_remote(0);
  sync->add_remote(1);
  sync->add_remote(2);
  sync->received(0, 0, 1, 0, 1);
  sync->received(2, 0, 1, 0, 1);
  sync->received(0, 0, 0, 0, 1);
  sync->received(0, 0, 2, 0, 1);
  sync->received(1, 0, 2, 0, 1);
  sync->received(2, 0, 0, 0, 1);
  sync->received(1, 0, 1, 0, 1);
  sync->received(1, 0, 0, 0, 1);
  sync->received(2, 0, 2, 0, 1);
}

TEST_F(SyncBlobInfoTest, TripleLayerSingleBlobSinglePartSingleRemoteSyncing) {
  shared_ptr<BlobSyncInfo> sync = prepare_const_mock(
          list_of<vector<int> >
          (list_of<int>(1))
          (list_of<int>(1))
          (list_of<int>(1)));
  {
    InSequence in_seq;
    EXPECT_CALL(*sync_mock, synced(1, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(1, 1));
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 1));
    EXPECT_CALL(*sync_mock, synced(2, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(2, 1));
    EXPECT_CALL(*sync_mock, synced(1));
  }
  sync->add_remote(0);
  sync->received(0, 1, 0, 0, 1);
  sync->received(0, 0, 0, 0, 1);
  sync->received(0, 2, 0, 0, 1);
}

TEST_F(SyncBlobInfoTest, TripleLayerSingleBlobSinglePartTripleRemoteSyncing) {
  shared_ptr<BlobSyncInfo> sync = prepare_const_mock(
          list_of<vector<int> >
          (list_of<int>(1))
          (list_of<int>(1))
          (list_of<int>(1)));
  {
    InSequence in_seq;
    EXPECT_CALL(*sync_mock, synced(1, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(1, 1));
    EXPECT_CALL(*sync_mock, synced(2, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(2, 1));
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 1));
    EXPECT_CALL(*sync_mock, synced(1));
  }
  sync->add_remote(8);
  sync->add_remote(9);
  sync->add_remote(7);
  sync->received(8, 1, 0, 0, 1);
  sync->received(9, 1, 0, 0, 1);
  sync->received(7, 1, 0, 0, 1);
  sync->received(9, 2, 0, 0, 1);
  sync->received(7, 2, 0, 0, 1);
  sync->received(8, 2, 0, 0, 1);
  sync->received(9, 0, 0, 0, 1);
  sync->received(8, 0, 0, 0, 1);
  sync->received(7, 0, 0, 0, 1);
}


TEST_F(SyncBlobInfoTest, TripleLayerTripleBlobTriplePartTripleRemoteSyncing) {
  shared_ptr<BlobSyncInfo> sync = prepare_const_mock(
    list_of<vector<int> >(list_of<int>(2))
                         (list_of<int>(2)(3))
                         (list_of<int>(1)(2)(3)));
  sync->add_remote(0);
  sync->add_remote(1);
  sync->add_remote(2);
  sync->received(0, 0, 0, 0, 1);
  sync->received(2, 0, 0, 1, 1);
  sync->received(1, 0, 0, 0, 1);
  sync->received(0, 1, 0, 0, 1);
  sync->received(1, 0, 0, 1, 1);
  sync->received(2, 2, 2, 1, 1);
  sync->received(0, 2, 0, 0, 1);
  sync->received(1, 1, 0, 0, 1);
  sync->received(2, 1, 1, 1, 1);
  EXPECT_CALL(*sync_mock, synced(0, 0, 1, 1));
  sync->received(0, 0, 0, 1, 1);
  Mock::VerifyAndClearExpectations(&sync_mock);

  sync->received(0, 1, 0, 1, 1);
  sync->received(0, 2, 1, 0, 1);
  sync->received(1, 1, 0, 1, 1);
  EXPECT_CALL(*sync_mock, synced(1, 0, 1, 1));
  sync->received(2, 1, 0, 1, 1);
  Mock::VerifyAndClearExpectations(&sync_mock);

  sync->received(1, 1, 1, 0, 1);
  sync->received(2, 2, 1, 0, 1);
  sync->received(2, 2, 1, 1, 1);
  sync->received(1, 1, 1, 1, 1);
  sync->received(1, 1, 1, 2, 1);
  sync->received(0, 1, 1, 0, 1);
  sync->received(0, 2, 1, 1, 1);
  sync->received(2, 2, 2, 2, 1);
  sync->received(1, 2, 0, 0, 1);
  EXPECT_CALL(*sync_mock, synced(2, 1, 0, 1));
  sync->received(1, 2, 1, 0, 1);
  Mock::VerifyAndClearExpectations(&sync_mock);

  EXPECT_CALL(*sync_mock, synced(1, 1, 1, 1));
  sync->received(0, 1, 1, 1, 1);
  Mock::VerifyAndClearExpectations(&sync_mock);

  sync->received(0, 2, 2, 0, 1);
  sync->received(2, 1, 1, 2, 1);
  EXPECT_CALL(*sync_mock, synced(2, 1, 1, 1));
  sync->received(1, 2, 1, 1, 1);
  Mock::VerifyAndClearExpectations(&sync_mock);

  EXPECT_CALL(*sync_mock, synced(2, 0, 0, 1));
  sync->received(2, 2, 0, 0, 1);
  Mock::VerifyAndClearExpectations(&sync_mock);


  EXPECT_CALL(*sync_mock, synced(1, 1, 2, 1));
  sync->received(0, 1, 1, 2, 1);
  Mock::VerifyAndClearExpectations(&sync_mock);


  sync->received(2, 2, 2, 0, 1);
  EXPECT_CALL(*sync_mock, synced(2, 2, 0, 1));
  sync->received(1, 2, 2, 0, 1);


  sync->received(0, 2, 2, 1, 1);
  EXPECT_CALL(*sync_mock, synced(1, 1, 0, 1));
  sync->received(2, 1, 1, 0, 1);
  Mock::VerifyAndClearExpectations(&sync_mock);


  EXPECT_CALL(*sync_mock, synced(2, 2, 1, 1));
  sync->received(1, 2, 2, 1, 1);
  Mock::VerifyAndClearExpectations(&sync_mock);

  {
    InSequence in_seq;
    EXPECT_CALL(*sync_mock, synced(1, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(1, 1));
  }
  sync->received(2, 1, 0, 0, 1);
  Mock::VerifyAndClearExpectations(&sync_mock);

  sync->received(0, 2, 2, 2, 1);

  {
    InSequence in_seq;
    EXPECT_CALL(*sync_mock, synced(2, 2, 2, 1));
    EXPECT_CALL(*sync_mock, synced(2, 1));
  }
  sync->received(1, 2, 2, 2, 1);
  Mock::VerifyAndClearExpectations(&sync_mock);

  {
    InSequence in_seq;
    EXPECT_CALL(*sync_mock, synced(0, 0, 0, 1));
    EXPECT_CALL(*sync_mock, synced(0, 1));
    EXPECT_CALL(*sync_mock, synced(1));
  }
  sync->received(2, 0, 0, 0, 1);
  Mock::VerifyAndClearExpectations(&sync_mock);
}

}  // namespace
}  // namespace caffe
