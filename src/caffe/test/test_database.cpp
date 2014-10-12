#include <string>
#include <vector>

#include "caffe/util/io.hpp"

#include "gtest/gtest.h"

#include "caffe/database_factory.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class DatabaseTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  string DBName() {
    string filename;
    MakeTempDir(&filename);
    filename += "/db";
    return filename;
  }

  Database::buffer_t TestKey() {
    const char* kKey = "hello";
    Database::buffer_t key(kKey, kKey + 5);
    return key;
  }

  Database::buffer_t TestValue() {
    const char* kValue = "world";
    Database::buffer_t value(kValue, kValue + 5);
    return value;
  }
};

TYPED_TEST_CASE(DatabaseTest, TestDtypesAndDevices);

TYPED_TEST(DatabaseTest, TestNewDoesntExistLevelDBPasses) {
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(this->DBName(), Database::New);
  database->close();
}

TYPED_TEST(DatabaseTest, TestNewExistsFailsLevelDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::New);
  database->close();

  EXPECT_DEATH(database->open(name, Database::New), "");
}

TYPED_TEST(DatabaseTest, TestReadOnlyExistsLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::New);
  database->close();

  database->open(name, Database::ReadOnly);
  database->close();
}

TYPED_TEST(DatabaseTest, TestReadOnlyDoesntExistFailsLevelDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_DEATH(database->open(name, Database::ReadOnly), "");
}

TYPED_TEST(DatabaseTest, TestReadWriteExistsLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::New);
  database->close();

  database->open(name, Database::ReadWrite);
  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteDoesntExistLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::ReadWrite);
  database->close();
}

TYPED_TEST(DatabaseTest, TestIteratorsLevelDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::New);

  const int kNumExamples = 4;
  for (int i = 0; i < kNumExamples; ++i) {
    stringstream ss;
    ss << i;
    string key = ss.str();
    ss << " here be data";
    string value = ss.str();
    Database::buffer_t key_buf(key.data(), key.data() + key.size());
    Database::buffer_t val_buf(value.data(), value.data() + value.size());
    database->put(&key_buf, &val_buf);
  }
  database->commit();

  int count = 0;
  for (Database::const_iterator iter = database->begin();
      iter != database->end(); ++iter) {
    (void)iter;
    ++count;
  }

  EXPECT_EQ(kNumExamples, count);
}

TYPED_TEST(DatabaseTest, TestNewPutLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::New);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  database->commit();

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewCommitLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::New);

  database->commit();

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewGetLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::New);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  database->commit();

  Database::buffer_t new_val;

  database->get(&key, &new_val);

  EXPECT_EQ(val.size(), new_val.size());
  for (size_t i = 0; i < val.size(); ++i) {
    EXPECT_EQ(val.at(i), new_val.at(i));
  }

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewGetNoCommitLevelDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::New);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  Database::buffer_t new_val;

  EXPECT_DEATH(database->get(&key, &new_val), "");
}


TYPED_TEST(DatabaseTest, TestReadWritePutLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::ReadWrite);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  database->commit();

  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteCommitLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::ReadWrite);

  database->commit();

  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteGetLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::New);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  database->commit();

  Database::buffer_t new_val;

  database->get(&key, &new_val);

  EXPECT_EQ(val.size(), new_val.size());
  for (size_t i = 0; i < val.size(); ++i) {
    EXPECT_EQ(val.at(i), new_val.at(i));
  }

  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteGetNoCommitLevelDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::New);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  Database::buffer_t new_val;

  EXPECT_DEATH(database->get(&key, &new_val), "");
}

TYPED_TEST(DatabaseTest, TestReadOnlyPutLevelDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::New);
  database->close();

  database->open(name, Database::ReadOnly);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_DEATH(database->put(&key, &val), "");
}

TYPED_TEST(DatabaseTest, TestReadOnlyCommitLevelDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::New);
  database->close();

  database->open(name, Database::ReadOnly);

  EXPECT_DEATH(database->commit(), "");
}

TYPED_TEST(DatabaseTest, TestReadOnlyGetLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::New);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  database->commit();

  database->close();

  database->open(name, Database::ReadOnly);

  Database::buffer_t new_val;

  database->get(&key, &new_val);

  EXPECT_EQ(val.size(), new_val.size());
  for (size_t i = 0; i < val.size(); ++i) {
    EXPECT_EQ(val.at(i), new_val.at(i));
  }
}

TYPED_TEST(DatabaseTest, TestReadOnlyGetNoCommitLevelDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  database->open(name, Database::New);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  database->close();

  database->open(name, Database::ReadOnly);

  Database::buffer_t new_val;

  EXPECT_DEATH(database->get(&key, &new_val), "");
}

TYPED_TEST(DatabaseTest, TestNewDoesntExistLMDBPasses) {
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(this->DBName(), Database::New);
  database->close();
}

TYPED_TEST(DatabaseTest, TestNewExistsFailsLMDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::New);
  database->close();

  EXPECT_DEATH(database->open(name, Database::New), "");
}

TYPED_TEST(DatabaseTest, TestReadOnlyExistsLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::New);
  database->close();

  database->open(name, Database::ReadOnly);
  database->close();
}

TYPED_TEST(DatabaseTest, TestReadOnlyDoesntExistFailsLMDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_DEATH(database->open(name, Database::ReadOnly), "");
}

TYPED_TEST(DatabaseTest, TestReadWriteExistsLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::New);
  database->close();

  database->open(name, Database::ReadWrite);
  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteDoesntExistLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::ReadWrite);
  database->close();
}

TYPED_TEST(DatabaseTest, TestIteratorsLMDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::New);

  const int kNumExamples = 4;
  for (int i = 0; i < kNumExamples; ++i) {
    stringstream ss;
    ss << i;
    string key = ss.str();
    ss << " here be data";
    string value = ss.str();
    Database::buffer_t key_buf(key.data(), key.data() + key.size());
    Database::buffer_t val_buf(value.data(), value.data() + value.size());
    database->put(&key_buf, &val_buf);
  }
  database->commit();

  int count = 0;
  for (Database::const_iterator iter = database->begin();
      iter != database->end(); ++iter) {
    (void)iter;
    ++count;
  }

  EXPECT_EQ(kNumExamples, count);
}

TYPED_TEST(DatabaseTest, TestNewPutLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::New);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  database->commit();

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewCommitLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::New);

  database->commit();

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewGetLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::New);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  database->commit();

  Database::buffer_t new_val;

  database->get(&key, &new_val);

  EXPECT_EQ(val.size(), new_val.size());
  for (size_t i = 0; i < val.size(); ++i) {
    EXPECT_EQ(val.at(i), new_val.at(i));
  }

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewGetNoCommitLMDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::New);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  Database::buffer_t new_val;

  EXPECT_DEATH(database->get(&key, &new_val), "");
}

TYPED_TEST(DatabaseTest, TestReadWritePutLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::ReadWrite);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  database->commit();

  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteCommitLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::ReadWrite);

  database->commit();

  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteGetLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::New);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  database->commit();

  Database::buffer_t new_val;

  database->get(&key, &new_val);

  EXPECT_EQ(val.size(), new_val.size());
  for (size_t i = 0; i < val.size(); ++i) {
    EXPECT_EQ(val.at(i), new_val.at(i));
  }

  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteGetNoCommitLMDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::New);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  Database::buffer_t new_val;

  EXPECT_DEATH(database->get(&key, &new_val), "");
}

TYPED_TEST(DatabaseTest, TestReadOnlyPutLMDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::New);
  database->close();

  database->open(name, Database::ReadOnly);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_DEATH(database->put(&key, &val), "");
}

TYPED_TEST(DatabaseTest, TestReadOnlyCommitLMDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::New);
  database->close();

  database->open(name, Database::ReadOnly);

  EXPECT_DEATH(database->commit(), "");
}

TYPED_TEST(DatabaseTest, TestReadOnlyGetLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::New);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  database->commit();

  database->close();

  database->open(name, Database::ReadOnly);

  Database::buffer_t new_val;

  database->get(&key, &new_val);

  EXPECT_EQ(val.size(), new_val.size());
  for (size_t i = 0; i < val.size(); ++i) {
    EXPECT_EQ(val.at(i), new_val.at(i));
  }
}

TYPED_TEST(DatabaseTest, TestReadOnlyGetNoCommitLMDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  database->open(name, Database::New);

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  database->put(&key, &val);

  database->close();

  database->open(name, Database::ReadOnly);

  Database::buffer_t new_val;

  EXPECT_DEATH(database->get(&key, &new_val), "");
}

}  // namespace caffe
