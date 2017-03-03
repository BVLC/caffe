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

#if defined(USE_LEVELDB) && defined(USE_LMDB) && defined(USE_OPENCV)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

using boost::scoped_ptr;

template <typename TypeParam>
class DBTest : public ::testing::Test {
 protected:
  DBTest()
      : backend_(TypeParam::backend),
      root_images_(string(EXAMPLES_SOURCE_DIR) + string("images/")) {}

  virtual void SetUp() {
    MakeTempDir(&source_);
    source_ += "/db";
    string keys[] = {"cat.jpg", "fish-bike.jpg"};
    LOG(INFO) << "Using temporary db " << source_;
    scoped_ptr<db::DB> db(db::GetDB(TypeParam::backend));
    db->Open(this->source_, db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());
    for (int i = 0; i < 2; ++i) {
      Datum datum;
      ReadImageToDatum(root_images_ + keys[i], i, &datum);
      string out;
      CHECK(datum.SerializeToString(&out));
      txn->Put(keys[i], out);
    }
    txn->Commit();
  }

  virtual ~DBTest() { }

  DataParameter_DB backend_;
  string source_;
  string root_images_;
};

struct TypeLevelDB {
  static DataParameter_DB backend;
};
DataParameter_DB TypeLevelDB::backend = DataParameter_DB_LEVELDB;

struct TypeLMDB {
  static DataParameter_DB backend;
};
DataParameter_DB TypeLMDB::backend = DataParameter_DB_LMDB;

// typedef ::testing::Types<TypeLmdb> TestTypes;
typedef ::testing::Types<TypeLevelDB, TypeLMDB> TestTypes;

TYPED_TEST_CASE(DBTest, TestTypes);

TYPED_TEST(DBTest, TestGetDB) {
  scoped_ptr<db::DB> db(db::GetDB(TypeParam::backend));
}

TYPED_TEST(DBTest, TestNext) {
  scoped_ptr<db::DB> db(db::GetDB(TypeParam::backend));
  db->Open(this->source_, db::READ);
  scoped_ptr<db::Cursor> cursor(db->NewCursor());
  EXPECT_TRUE(cursor->valid());
  cursor->Next();
  EXPECT_TRUE(cursor->valid());
  cursor->Next();
  EXPECT_FALSE(cursor->valid());
}

TYPED_TEST(DBTest, TestSeekToFirst) {
  scoped_ptr<db::DB> db(db::GetDB(TypeParam::backend));
  db->Open(this->source_, db::READ);
  scoped_ptr<db::Cursor> cursor(db->NewCursor());
  cursor->Next();
  cursor->SeekToFirst();
  EXPECT_TRUE(cursor->valid());
  string key = cursor->key();
  Datum datum;
  datum.ParseFromString(cursor->value());
  EXPECT_EQ(key, "cat.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
}

TYPED_TEST(DBTest, TestKeyValue) {
  scoped_ptr<db::DB> db(db::GetDB(TypeParam::backend));
  db->Open(this->source_, db::READ);
  scoped_ptr<db::Cursor> cursor(db->NewCursor());
  EXPECT_TRUE(cursor->valid());
  string key = cursor->key();
  Datum datum;
  datum.ParseFromString(cursor->value());
  EXPECT_EQ(key, "cat.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
  cursor->Next();
  EXPECT_TRUE(cursor->valid());
  key = cursor->key();
  datum.ParseFromString(cursor->value());
  EXPECT_EQ(key, "fish-bike.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 323);
  EXPECT_EQ(datum.width(), 481);
  cursor->Next();
  EXPECT_FALSE(cursor->valid());
}

TYPED_TEST(DBTest, TestWrite) {
  scoped_ptr<db::DB> db(db::GetDB(TypeParam::backend));
  db->Open(this->source_, db::WRITE);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());
  Datum datum;
  ReadFileToDatum(this->root_images_ + "cat.jpg", 0, &datum);
  string out;
  CHECK(datum.SerializeToString(&out));
  txn->Put("cat.jpg", out);
  ReadFileToDatum(this->root_images_ + "fish-bike.jpg", 1, &datum);
  CHECK(datum.SerializeToString(&out));
  txn->Put("fish-bike.jpg", out);
  txn->Commit();
}

}  // namespace caffe
#endif  // USE_LEVELDB, USE_LMDB and USE_OPENCV
