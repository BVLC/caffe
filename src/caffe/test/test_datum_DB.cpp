#include <string>

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/datum_DB.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class DatumDBTest : public ::testing::Test {
 protected:
  DatumDBTest()
      : backend_(TypeParam::backend),
      root_images_(string(EXAMPLES_SOURCE_DIR) + string("images/")) {}

  virtual void SetUp() {
    MakeTempDir(&source_);
    source_ += "/db";
    string keys[] = {"cat.jpg", "fish-bike.jpg"};
    LOG(INFO) << "Using temporary datumdb " << source_;
    DatumDBParameter param;
    param.set_backend(backend_);
    param.set_mode(DatumDBParameter_Mode_NEW);
    param.set_source(source_);
    param.set_root_images(root_images_);
    shared_ptr<DatumDB> datumdb = DatumDB::GetDatumDB(param);
    for (int i = 0; i < 2; ++i) {
      Datum datum;
      ReadImageToDatum(root_images_ + keys[i], i, &datum);
      datumdb->Put(keys[i], datum);
    }
    datumdb->Commit();
  }

  virtual ~DatumDBTest() { }

  DatumDBParameter_Backend backend_;
  string source_;
  string root_images_;
};

struct TypeLeveldb {
  static const DatumDBParameter_Backend backend;
};
const DatumDBParameter_Backend TypeLeveldb::backend =
      DatumDBParameter_Backend_LEVELDB;

struct TypeLmdb {
  static const DatumDBParameter_Backend backend;
};
const DatumDBParameter_Backend TypeLmdb::backend =
      DatumDBParameter_Backend_LMDB;

struct TypeImagesdb {
  static const DatumDBParameter_Backend backend;
};
const DatumDBParameter_Backend TypeImagesdb::backend =
      DatumDBParameter_Backend_IMAGESDB;

// typedef ::testing::Types<TypeLmdb> TestTypes;
typedef ::testing::Types<TypeLeveldb, TypeLmdb, TypeImagesdb> TestTypes;

TYPED_TEST_CASE(DatumDBTest, TestTypes);

TYPED_TEST(DatumDBTest, TestGetDatumDB) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  shared_ptr<DatumDB> datumdb = DatumDB::GetDatumDB(param);
  EXPECT_TRUE(datumdb->Valid());
}

TYPED_TEST(DatumDBTest, TestNext) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  shared_ptr<DatumDB> datumdb = DatumDB::GetDatumDB(param);
  EXPECT_TRUE(datumdb->Valid());
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(datumdb->Next());
  }
}

TYPED_TEST(DatumDBTest, TestNextNoLoop) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  param.set_loop(false);
  shared_ptr<DatumDB> datumdb = DatumDB::GetDatumDB(param);
  EXPECT_TRUE(datumdb->Valid());
  EXPECT_TRUE(datumdb->Next());
  EXPECT_FALSE(datumdb->Next());
}

TYPED_TEST(DatumDBTest, TestPrev) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  shared_ptr<DatumDB> datumdb = DatumDB::GetDatumDB(param);
  EXPECT_TRUE(datumdb->Valid());
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(datumdb->Prev());
  }
}

TYPED_TEST(DatumDBTest, TestPrevNoLoop) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  param.set_loop(false);
  shared_ptr<DatumDB> datumdb = DatumDB::GetDatumDB(param);
  EXPECT_TRUE(datumdb->Valid());
  EXPECT_FALSE(datumdb->Prev());
}

TYPED_TEST(DatumDBTest, TestSeekToFirst) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  param.set_loop(false);
  shared_ptr<DatumDB> datumdb = DatumDB::GetDatumDB(param);
  EXPECT_TRUE(datumdb->SeekToFirst());
  string key;
  Datum datum;
  EXPECT_TRUE(datumdb->Current(&key, &datum));
  EXPECT_TRUE(key == "cat.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
  EXPECT_FALSE(datumdb->Prev());
}

TYPED_TEST(DatumDBTest, TestSeekToLast) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  param.set_loop(false);
  shared_ptr<DatumDB> datumdb = DatumDB::GetDatumDB(param);
  EXPECT_TRUE(datumdb->SeekToLast());
  string key;
  Datum datum;
  EXPECT_TRUE(datumdb->Current(&key, &datum));
  EXPECT_TRUE(key == "fish-bike.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 323);
  EXPECT_EQ(datum.width(), 481);
  EXPECT_FALSE(datumdb->Next());
}

TYPED_TEST(DatumDBTest, TestCurrent) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  shared_ptr<DatumDB> datumdb = DatumDB::GetDatumDB(param);
  string key;
  Datum datum;
  EXPECT_TRUE(datumdb->Current(&key, &datum));
  EXPECT_TRUE(key == "cat.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
  EXPECT_TRUE(datumdb->Current(&key, &datum));
  EXPECT_TRUE(key == "cat.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
  EXPECT_TRUE(datumdb->Next());
  EXPECT_TRUE(datumdb->Current(&key, &datum));
  EXPECT_TRUE(key == "fish-bike.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 323);
  EXPECT_EQ(datum.width(), 481);
  EXPECT_TRUE(datumdb->Prev());
  EXPECT_TRUE(datumdb->Current(&key, &datum));
  EXPECT_TRUE(key == "cat.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
}

TYPED_TEST(DatumDBTest, TestGet) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  shared_ptr<DatumDB> datumdb = DatumDB::GetDatumDB(param);
  Datum datum;
  EXPECT_TRUE(datumdb->Get("cat.jpg", &datum));
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
  EXPECT_TRUE(datumdb->Get("fish-bike.jpg", &datum));
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 323);
  EXPECT_EQ(datum.width(), 481);
  EXPECT_FALSE(datumdb->Get("none.jpg", &datum));
}

TYPED_TEST(DatumDBTest, TestCurrentKey) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  shared_ptr<DatumDB> datumdb = DatumDB::GetDatumDB(param);
  string key;
  key = datumdb->CurrentKey();
  EXPECT_TRUE(key == "cat.jpg");
  EXPECT_TRUE(datumdb->Next());
  key = datumdb->CurrentKey();
  EXPECT_TRUE(key == "fish-bike.jpg");
}

TYPED_TEST(DatumDBTest, TestCurrentDatum) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  shared_ptr<DatumDB> datumdb = DatumDB::GetDatumDB(param);
  Datum datum;
  datum = datumdb->CurrentDatum();
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
  EXPECT_TRUE(datumdb->Next());
  datum = datumdb->CurrentDatum();
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 323);
  EXPECT_EQ(datum.width(), 481);
}

TYPED_TEST(DatumDBTest, TestWrite) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_mode(DatumDBParameter_Mode_WRITE);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  shared_ptr<DatumDB> datumdb = DatumDB::GetDatumDB(param);
  Datum datum;
  ReadFileToDatum(this->root_images_ + "cat.jpg", 0, &datum);
  datumdb->Put("cat.jpg", datum);
  ReadFileToDatum(this->root_images_ + "fish-bike.jpg", 1, &datum);
  datumdb->Put("fish-bike.jpg", datum);
  datumdb->Commit();
}

}  // namespace caffe
