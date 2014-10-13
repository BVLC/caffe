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

  Database::buffer_t TestAltKey() {
    const char* kKey = "foo";
    Database::buffer_t key(kKey, kKey + 3);
    return key;
  }

  Database::buffer_t TestAltValue() {
    const char* kValue = "bar";
    Database::buffer_t value(kValue, kValue + 3);
    return value;
  }

  bool BufferEq(const Database::buffer_t& buf1,
      const Database::buffer_t& buf2) {
    if (buf1.size() != buf2.size()) {
      return false;
    }
    for (size_t i = 0; i < buf1.size(); ++i) {
      if (buf1.at(i) != buf2.at(i)) {
        return false;
      }
    }

    return true;
  }
};

TYPED_TEST_CASE(DatabaseTest, TestDtypesAndDevices);

TYPED_TEST(DatabaseTest, TestNewDoesntExistLevelDBPasses) {
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(this->DBName(), Database::New));
  database->close();
}

TYPED_TEST(DatabaseTest, TestNewExistsFailsLevelDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));
  database->close();

  EXPECT_FALSE(database->open(name, Database::New));
}

TYPED_TEST(DatabaseTest, TestReadOnlyExistsLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));
  database->close();

  EXPECT_TRUE(database->open(name, Database::ReadOnly));
  database->close();
}

TYPED_TEST(DatabaseTest, TestReadOnlyDoesntExistFailsLevelDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_FALSE(database->open(name, Database::ReadOnly));
}

TYPED_TEST(DatabaseTest, TestReadWriteExistsLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));
  database->close();

  EXPECT_TRUE(database->open(name, Database::ReadWrite));
  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteDoesntExistLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::ReadWrite));
  database->close();
}

TYPED_TEST(DatabaseTest, TestIteratorsLevelDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));

  const int kNumExamples = 4;
  for (int i = 0; i < kNumExamples; ++i) {
    stringstream ss;
    ss << i;
    string key = ss.str();
    ss << " here be data";
    string value = ss.str();
    Database::buffer_t key_buf(key.data(), key.data() + key.size());
    Database::buffer_t val_buf(value.data(), value.data() + value.size());
    EXPECT_TRUE(database->put(&key_buf, &val_buf));
  }
  EXPECT_TRUE(database->commit());

  int count = 0;
  for (Database::const_iterator iter = database->begin();
      iter != database->end(); ++iter) {
    (void)iter;
    ++count;
  }

  EXPECT_EQ(kNumExamples, count);
}

TYPED_TEST(DatabaseTest, TestIteratorsPreIncrementLevelDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key1 = this->TestAltKey();
  Database::buffer_t value1 = this->TestAltValue();

  Database::buffer_t key2 = this->TestKey();
  Database::buffer_t value2 = this->TestValue();

  EXPECT_TRUE(database->put(&key1, &value1));
  EXPECT_TRUE(database->put(&key2, &value2));
  EXPECT_TRUE(database->commit());

  Database::const_iterator iter1 = database->begin();

  EXPECT_FALSE(database->end() == iter1);

  EXPECT_TRUE(this->BufferEq(iter1->key, key1));

  Database::const_iterator iter2 = ++iter1;

  EXPECT_FALSE(database->end() == iter1);
  EXPECT_FALSE(database->end() == iter2);

  EXPECT_TRUE(this->BufferEq(iter2->key, key2));

  Database::const_iterator iter3 = ++iter2;

  EXPECT_TRUE(database->end() == iter3);

  iter1 = database->end();
  iter2 = database->end();
  iter3 = database->end();

  database->close();
}

TYPED_TEST(DatabaseTest, TestIteratorsPostIncrementLevelDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key1 = this->TestAltKey();
  Database::buffer_t value1 = this->TestAltValue();

  Database::buffer_t key2 = this->TestKey();
  Database::buffer_t value2 = this->TestValue();

  EXPECT_TRUE(database->put(&key1, &value1));
  EXPECT_TRUE(database->put(&key2, &value2));
  EXPECT_TRUE(database->commit());

  Database::const_iterator iter1 = database->begin();

  EXPECT_FALSE(database->end() == iter1);

  EXPECT_TRUE(this->BufferEq(iter1->key, key1));

  Database::const_iterator iter2 = iter1++;

  EXPECT_FALSE(database->end() == iter1);
  EXPECT_FALSE(database->end() == iter2);

  EXPECT_TRUE(this->BufferEq(iter2->key, key1));
  EXPECT_TRUE(this->BufferEq(iter1->key, key2));

  Database::const_iterator iter3 = iter1++;

  EXPECT_FALSE(database->end() == iter3);
  EXPECT_TRUE(this->BufferEq(iter3->key, key2));
  EXPECT_TRUE(database->end() == iter1);

  iter1 = database->end();
  iter2 = database->end();
  iter3 = database->end();

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewPutLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  EXPECT_TRUE(database->commit());

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewCommitLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));

  EXPECT_TRUE(database->commit());

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewGetLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  EXPECT_TRUE(database->commit());

  Database::buffer_t new_val;

  EXPECT_TRUE(database->get(&key, &new_val));

  EXPECT_TRUE(this->BufferEq(val, new_val));

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewGetNoCommitLevelDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  Database::buffer_t new_val;

  EXPECT_FALSE(database->get(&key, &new_val));
}


TYPED_TEST(DatabaseTest, TestReadWritePutLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::ReadWrite));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  EXPECT_TRUE(database->commit());

  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteCommitLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::ReadWrite));

  EXPECT_TRUE(database->commit());

  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteGetLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  EXPECT_TRUE(database->commit());

  Database::buffer_t new_val;

  EXPECT_TRUE(database->get(&key, &new_val));

  EXPECT_TRUE(this->BufferEq(val, new_val));

  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteGetNoCommitLevelDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  Database::buffer_t new_val;

  EXPECT_FALSE(database->get(&key, &new_val));
}

TYPED_TEST(DatabaseTest, TestReadOnlyPutLevelDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));
  database->close();

  EXPECT_TRUE(database->open(name, Database::ReadOnly));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_FALSE(database->put(&key, &val));
}

TYPED_TEST(DatabaseTest, TestReadOnlyCommitLevelDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));
  database->close();

  EXPECT_TRUE(database->open(name, Database::ReadOnly));

  EXPECT_FALSE(database->commit());
}

TYPED_TEST(DatabaseTest, TestReadOnlyGetLevelDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  EXPECT_TRUE(database->commit());

  database->close();

  EXPECT_TRUE(database->open(name, Database::ReadOnly));

  Database::buffer_t new_val;

  EXPECT_TRUE(database->get(&key, &new_val));

  EXPECT_TRUE(this->BufferEq(val, new_val));
}

TYPED_TEST(DatabaseTest, TestReadOnlyGetNoCommitLevelDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("leveldb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  database->close();

  EXPECT_TRUE(database->open(name, Database::ReadOnly));

  Database::buffer_t new_val;

  EXPECT_FALSE(database->get(&key, &new_val));
}

TYPED_TEST(DatabaseTest, TestNewDoesntExistLMDBPasses) {
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(this->DBName(), Database::New));
  database->close();
}

TYPED_TEST(DatabaseTest, TestNewExistsFailsLMDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));
  database->close();

  EXPECT_FALSE(database->open(name, Database::New));
}

TYPED_TEST(DatabaseTest, TestReadOnlyExistsLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));
  database->close();

  EXPECT_TRUE(database->open(name, Database::ReadOnly));
  database->close();
}

TYPED_TEST(DatabaseTest, TestReadOnlyDoesntExistFailsLMDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_FALSE(database->open(name, Database::ReadOnly));
}

TYPED_TEST(DatabaseTest, TestReadWriteExistsLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));
  database->close();

  EXPECT_TRUE(database->open(name, Database::ReadWrite));
  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteDoesntExistLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::ReadWrite));
  database->close();
}

TYPED_TEST(DatabaseTest, TestIteratorsLMDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));

  const int kNumExamples = 4;
  for (int i = 0; i < kNumExamples; ++i) {
    stringstream ss;
    ss << i;
    string key = ss.str();
    ss << " here be data";
    string value = ss.str();
    Database::buffer_t key_buf(key.data(), key.data() + key.size());
    Database::buffer_t val_buf(value.data(), value.data() + value.size());
    EXPECT_TRUE(database->put(&key_buf, &val_buf));
  }
  EXPECT_TRUE(database->commit());

  int count = 0;
  for (Database::const_iterator iter = database->begin();
      iter != database->end(); ++iter) {
    (void)iter;
    ++count;
  }

  EXPECT_EQ(kNumExamples, count);
}

TYPED_TEST(DatabaseTest, TestIteratorsPreIncrementLMDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key1 = this->TestAltKey();
  Database::buffer_t value1 = this->TestAltValue();

  Database::buffer_t key2 = this->TestKey();
  Database::buffer_t value2 = this->TestValue();

  EXPECT_TRUE(database->put(&key1, &value1));
  EXPECT_TRUE(database->put(&key2, &value2));
  EXPECT_TRUE(database->commit());

  Database::const_iterator iter1 = database->begin();

  EXPECT_FALSE(database->end() == iter1);

  EXPECT_TRUE(this->BufferEq(iter1->key, key1));

  Database::const_iterator iter2 = ++iter1;

  EXPECT_FALSE(database->end() == iter1);
  EXPECT_FALSE(database->end() == iter2);

  EXPECT_TRUE(this->BufferEq(iter2->key, key2));

  Database::const_iterator iter3 = ++iter2;

  EXPECT_TRUE(database->end() == iter3);

  iter1 = database->end();
  iter2 = database->end();
  iter3 = database->end();

  database->close();
}

TYPED_TEST(DatabaseTest, TestIteratorsPostIncrementLMDB) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key1 = this->TestAltKey();
  Database::buffer_t value1 = this->TestAltValue();

  Database::buffer_t key2 = this->TestKey();
  Database::buffer_t value2 = this->TestValue();

  EXPECT_TRUE(database->put(&key1, &value1));
  EXPECT_TRUE(database->put(&key2, &value2));
  EXPECT_TRUE(database->commit());

  Database::const_iterator iter1 = database->begin();

  EXPECT_FALSE(database->end() == iter1);

  EXPECT_TRUE(this->BufferEq(iter1->key, key1));

  Database::const_iterator iter2 = iter1++;

  EXPECT_FALSE(database->end() == iter1);
  EXPECT_FALSE(database->end() == iter2);

  EXPECT_TRUE(this->BufferEq(iter2->key, key1));
  EXPECT_TRUE(this->BufferEq(iter1->key, key2));

  Database::const_iterator iter3 = iter1++;

  EXPECT_FALSE(database->end() == iter3);
  EXPECT_TRUE(this->BufferEq(iter3->key, key2));
  EXPECT_TRUE(database->end() == iter1);

  iter1 = database->end();
  iter2 = database->end();
  iter3 = database->end();

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewPutLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  EXPECT_TRUE(database->commit());

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewCommitLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));

  EXPECT_TRUE(database->commit());

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewGetLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  EXPECT_TRUE(database->commit());

  Database::buffer_t new_val;

  EXPECT_TRUE(database->get(&key, &new_val));

  EXPECT_TRUE(this->BufferEq(val, new_val));

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewGetNoCommitLMDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  Database::buffer_t new_val;

  EXPECT_FALSE(database->get(&key, &new_val));
}

TYPED_TEST(DatabaseTest, TestReadWritePutLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::ReadWrite));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  EXPECT_TRUE(database->commit());

  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteCommitLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::ReadWrite));

  EXPECT_TRUE(database->commit());

  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteGetLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  EXPECT_TRUE(database->commit());

  Database::buffer_t new_val;

  EXPECT_TRUE(database->get(&key, &new_val));

  EXPECT_TRUE(this->BufferEq(val, new_val));

  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteGetNoCommitLMDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  Database::buffer_t new_val;

  EXPECT_FALSE(database->get(&key, &new_val));
}

TYPED_TEST(DatabaseTest, TestReadOnlyPutLMDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));
  database->close();

  EXPECT_TRUE(database->open(name, Database::ReadOnly));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_FALSE(database->put(&key, &val));
}

TYPED_TEST(DatabaseTest, TestReadOnlyCommitLMDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));
  database->close();

  EXPECT_TRUE(database->open(name, Database::ReadOnly));

  EXPECT_FALSE(database->commit());
}

TYPED_TEST(DatabaseTest, TestReadOnlyGetLMDBPasses) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  EXPECT_TRUE(database->commit());

  database->close();

  EXPECT_TRUE(database->open(name, Database::ReadOnly));

  Database::buffer_t new_val;

  EXPECT_TRUE(database->get(&key, &new_val));

  EXPECT_TRUE(this->BufferEq(val, new_val));
}

TYPED_TEST(DatabaseTest, TestReadOnlyGetNoCommitLMDBFails) {
  string name = this->DBName();
  shared_ptr<Database> database = DatabaseFactory("lmdb");
  EXPECT_TRUE(database->open(name, Database::New));

  Database::buffer_t key = this->TestKey();
  Database::buffer_t val = this->TestValue();

  EXPECT_TRUE(database->put(&key, &val));

  database->close();

  EXPECT_TRUE(database->open(name, Database::ReadOnly));

  Database::buffer_t new_val;

  EXPECT_FALSE(database->get(&key, &new_val));
}

}  // namespace caffe
