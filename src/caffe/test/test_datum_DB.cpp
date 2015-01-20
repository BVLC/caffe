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
    shared_ptr<DatumDB> datumdb(DatumDBRegistry::GetDatumDB(param));
    for (int i = 0; i < 2; ++i) {
      Datum datum;
      ReadImageToDatum(root_images_ + keys[i], i, &datum);
      datumdb->Put(keys[i], datum);
    }
    datumdb->Commit();
  }

  virtual ~DatumDBTest() { }

  string backend_;
  string source_;
  string root_images_;
};

struct TypeLeveldb {
  static const char backend[];
};
const char TypeLeveldb::backend[] = "leveldb";

struct TypeLmdb {
  static const char backend[];
};
const char TypeLmdb::backend[] = "lmdb";

struct TypeImagesdb {
  static const char backend[];
};
const char TypeImagesdb::backend[] = "imagesdb";

// typedef ::testing::Types<TypeLmdb> TestTypes;
typedef ::testing::Types<TypeLeveldb, TypeLmdb, TypeImagesdb> TestTypes;

TYPED_TEST_CASE(DatumDBTest, TestTypes);

TYPED_TEST(DatumDBTest, TestGetDatumDB) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  shared_ptr<DatumDB> datumdb(DatumDBRegistry::GetDatumDB(param));
  EXPECT_TRUE(datumdb.get());
}

TYPED_TEST(DatumDBTest, TestNext) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  shared_ptr<DatumDB> datumdb(DatumDBRegistry::GetDatumDB(param));
  shared_ptr<DatumDBCursor> datum_cursor(datumdb->NewCursor());
  EXPECT_TRUE(datum_cursor->Valid());
  for (int i = 0; i < 5; ++i) {
    datum_cursor->Next();
    EXPECT_TRUE(datum_cursor->Valid());
  }
}

TYPED_TEST(DatumDBTest, TestNextNoLoop) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  param.set_loop(false);
  shared_ptr<DatumDB> datumdb(DatumDBRegistry::GetDatumDB(param));
  shared_ptr<DatumDBCursor> datum_cursor(datumdb->NewCursor());
  EXPECT_TRUE(datum_cursor->Valid());
  datum_cursor->Next();
  EXPECT_TRUE(datum_cursor->Valid());
  datum_cursor->Next();
  EXPECT_FALSE(datum_cursor->Valid());
}

TYPED_TEST(DatumDBTest, TestSeekToFirst) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  param.set_loop(false);
  shared_ptr<DatumDB> datumdb(DatumDBRegistry::GetDatumDB(param));
  shared_ptr<DatumDBCursor> datum_cursor(datumdb->NewCursor());
  datum_cursor->SeekToFirst();
  EXPECT_TRUE(datum_cursor->Valid());
  string key = datum_cursor->key();
  Datum datum = datum_cursor->value();
  EXPECT_EQ(key, "cat.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
}

TYPED_TEST(DatumDBTest, TestKeyValue) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  shared_ptr<DatumDB> datumdb(DatumDBRegistry::GetDatumDB(param));
  shared_ptr<DatumDBCursor> datum_cursor(datumdb->NewCursor());
  EXPECT_TRUE(datum_cursor->Valid());
  string key = datum_cursor->key();
  Datum datum = datum_cursor->value();
  EXPECT_EQ(key, "cat.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
  EXPECT_TRUE(datum_cursor->Valid());
  key = datum_cursor->key();
  datum = datum_cursor->value();
  EXPECT_EQ(key, "cat.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
  datum_cursor->Next();
  EXPECT_TRUE(datum_cursor->Valid());
  key = datum_cursor->key();
  datum = datum_cursor->value();
  EXPECT_EQ(key, "fish-bike.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 323);
  EXPECT_EQ(datum.width(), 481);
  datum_cursor->Next();
  EXPECT_TRUE(datum_cursor->Valid());
  key = datum_cursor->key();
  datum = datum_cursor->value();
  EXPECT_EQ(key, "cat.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
}

TYPED_TEST(DatumDBTest, TestGet) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  shared_ptr<DatumDB> datumdb(DatumDBRegistry::GetDatumDB(param));
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

TYPED_TEST(DatumDBTest, TestWrite) {
  DatumDBParameter param;
  param.set_backend(TypeParam::backend);
  param.set_mode(DatumDBParameter_Mode_WRITE);
  param.set_source(this->source_);
  param.set_root_images(this->root_images_);
  shared_ptr<DatumDB> datumdb(DatumDBRegistry::GetDatumDB(param));
  Datum datum;
  ReadFileToDatum(this->root_images_ + "cat.jpg", 0, &datum);
  datumdb->Put("cat.jpg", datum);
  ReadFileToDatum(this->root_images_ + "fish-bike.jpg", 1, &datum);
  datumdb->Put("fish-bike.jpg", datum);
  datumdb->Commit();
}

}  // namespace caffe
