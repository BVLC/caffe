#include <string>
#include <vector>

#include "caffe/util/io.hpp"

#include "gtest/gtest.h"

#include "caffe/database_factory.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

namespace DatabaseTest_internal {

template <typename T>
struct TestData {
  static T TestValue();
  static T TestAltValue();
  static bool equals(const T& a, const T& b);
};

template <>
string TestData<string>::TestValue() {
  return "world";
}

template <>
string TestData<string>::TestAltValue() {
  return "bar";
}

template <>
bool TestData<string>::equals(const string& a, const string& b) {
  return a == b;
}

template <>
vector<char> TestData<vector<char> >::TestValue() {
  string str = "world";
  vector<char> val(str.data(), str.data() + str.size());
  return val;
}

template <>
vector<char> TestData<vector<char> >::TestAltValue() {
  string str = "bar";
  vector<char> val(str.data(), str.data() + str.size());
  return val;
}

template <>
bool TestData<vector<char> >::equals(const vector<char>& a,
    const vector<char>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a.at(i) != b.at(i)) {
      return false;
    }
  }

  return true;
}

template <>
Datum TestData<Datum>::TestValue() {
  Datum datum;
  datum.set_channels(3);
  datum.set_height(32);
  datum.set_width(32);
  datum.set_data(string(32 * 32 * 3 * 4, ' '));
  datum.set_label(0);
  return datum;
}

template <>
Datum TestData<Datum>::TestAltValue() {
  Datum datum;
  datum.set_channels(1);
  datum.set_height(64);
  datum.set_width(64);
  datum.set_data(string(64 * 64 * 1 * 4, ' '));
  datum.set_label(1);
  return datum;
}

template <>
bool TestData<Datum>::equals(const Datum& a, const Datum& b) {
  string serialized_a;
  a.SerializeToString(&serialized_a);

  string serialized_b;
  b.SerializeToString(&serialized_b);

  return serialized_a == serialized_b;
}

}  // namespace DatabaseTest_internal

#define UNPACK_TYPES \
  typedef typename TypeParam::value_type value_type; \
  const DataParameter_DB backend = TypeParam::backend;

template <typename TypeParam>
class DatabaseTest : public ::testing::Test {
 protected:
  typedef typename TypeParam::value_type value_type;

  string DBName() {
    string filename;
    MakeTempDir(&filename);
    filename += "/db";
    return filename;
  }

  string TestKey() {
    return "hello";
  }

  value_type TestValue() {
    return DatabaseTest_internal::TestData<value_type>::TestValue();
  }

  string TestAltKey() {
    return "foo";
  }

  value_type TestAltValue() {
    return DatabaseTest_internal::TestData<value_type>::TestAltValue();
  }

  template <typename T>
  bool equals(const T& a, const T& b) {
    return DatabaseTest_internal::TestData<T>::equals(a, b);
  }
};

struct StringLeveldb {
  typedef string value_type;
  static const DataParameter_DB backend;
};
const DataParameter_DB StringLeveldb::backend = DataParameter_DB_LEVELDB;

struct StringLmdb {
  typedef string value_type;
  static const DataParameter_DB backend;
};
const DataParameter_DB StringLmdb::backend = DataParameter_DB_LEVELDB;

struct VectorLeveldb {
  typedef vector<char> value_type;
  static const DataParameter_DB backend;
};
const DataParameter_DB VectorLeveldb::backend = DataParameter_DB_LEVELDB;

struct VectorLmdb {
  typedef vector<char> value_type;
  static const DataParameter_DB backend;
};
const DataParameter_DB VectorLmdb::backend = DataParameter_DB_LEVELDB;

struct DatumLeveldb {
  typedef Datum value_type;
  static const DataParameter_DB backend;
};
const DataParameter_DB DatumLeveldb::backend = DataParameter_DB_LEVELDB;

struct DatumLmdb {
  typedef Datum value_type;
  static const DataParameter_DB backend;
};
const DataParameter_DB DatumLmdb::backend = DataParameter_DB_LEVELDB;

typedef ::testing::Types<StringLeveldb, StringLmdb, VectorLeveldb, VectorLmdb,
    DatumLeveldb, DatumLmdb> TestTypes;

TYPED_TEST_CASE(DatabaseTest, TestTypes);

TYPED_TEST(DatabaseTest, TestNewDoesntExistPasses) {
  UNPACK_TYPES;

  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(this->DBName(),
      Database<string, value_type>::New));
  database->close();
}

TYPED_TEST(DatabaseTest, TestNewExistsFails) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));
  database->close();

  EXPECT_FALSE(database->open(name, Database<string, value_type>::New));
}

TYPED_TEST(DatabaseTest, TestReadOnlyExistsPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));
  database->close();

  EXPECT_TRUE(database->open(name, Database<string, value_type>::ReadOnly));
  database->close();
}

TYPED_TEST(DatabaseTest, TestReadOnlyDoesntExistFails) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_FALSE(database->open(name, Database<string, value_type>::ReadOnly));
}

TYPED_TEST(DatabaseTest, TestReadWriteExistsPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));
  database->close();

  EXPECT_TRUE(database->open(name, Database<string, value_type>::ReadWrite));
  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteDoesntExistPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::ReadWrite));
  database->close();
}

TYPED_TEST(DatabaseTest, TestKeys) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));

  string key1 = this->TestKey();
  value_type value1 = this->TestValue();

  EXPECT_TRUE(database->put(key1, value1));

  string key2 = this->TestAltKey();
  value_type value2 = this->TestAltValue();

  EXPECT_TRUE(database->put(key2, value2));

  EXPECT_TRUE(database->commit());

  vector<string> keys;
  database->keys(&keys);

  EXPECT_EQ(2, keys.size());

  EXPECT_TRUE(this->equals(keys.at(0), key1) ||
      this->equals(keys.at(0), key2));
  EXPECT_TRUE(this->equals(keys.at(1), key1) ||
      this->equals(keys.at(2), key2));
  EXPECT_FALSE(this->equals(keys.at(0), keys.at(1)));
}

TYPED_TEST(DatabaseTest, TestKeysNoCommit) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));

  string key1 = this->TestKey();
  value_type value1 = this->TestValue();

  EXPECT_TRUE(database->put(key1, value1));

  string key2 = this->TestAltKey();
  value_type value2 = this->TestAltValue();

  EXPECT_TRUE(database->put(key2, value2));

  vector<string> keys;
  database->keys(&keys);

  EXPECT_EQ(0, keys.size());
}

TYPED_TEST(DatabaseTest, TestIterators) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));

  const int kNumExamples = 4;
  for (int i = 0; i < kNumExamples; ++i) {
    stringstream ss;
    ss << i;
    string key = ss.str();
    ss << " here be data";
    value_type value = this->TestValue();
    EXPECT_TRUE(database->put(key, value));
  }
  EXPECT_TRUE(database->commit());

  int count = 0;
  typedef typename Database<string, value_type>::const_iterator Iter;
  for (Iter iter = database->begin(); iter != database->end(); ++iter) {
    (void)iter;
    ++count;
  }

  EXPECT_EQ(kNumExamples, count);
}

TYPED_TEST(DatabaseTest, TestIteratorsPreIncrement) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));

  string key1 = this->TestAltKey();
  value_type value1 = this->TestAltValue();

  string key2 = this->TestKey();
  value_type value2 = this->TestValue();

  EXPECT_TRUE(database->put(key1, value1));
  EXPECT_TRUE(database->put(key2, value2));
  EXPECT_TRUE(database->commit());

  typename Database<string, value_type>::const_iterator iter1 =
      database->begin();

  EXPECT_FALSE(database->end() == iter1);

  EXPECT_TRUE(this->equals(iter1->key, key1));

  typename Database<string, value_type>::const_iterator iter2 = ++iter1;

  EXPECT_FALSE(database->end() == iter1);
  EXPECT_FALSE(database->end() == iter2);

  EXPECT_TRUE(this->equals(iter2->key, key2));

  typename Database<string, value_type>::const_iterator iter3 = ++iter2;

  EXPECT_TRUE(database->end() == iter3);

  database->close();
}

TYPED_TEST(DatabaseTest, TestIteratorsPostIncrement) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));

  string key1 = this->TestAltKey();
  value_type value1 = this->TestAltValue();

  string key2 = this->TestKey();
  value_type value2 = this->TestValue();

  EXPECT_TRUE(database->put(key1, value1));
  EXPECT_TRUE(database->put(key2, value2));
  EXPECT_TRUE(database->commit());

  typename Database<string, value_type>::const_iterator iter1 =
      database->begin();

  EXPECT_FALSE(database->end() == iter1);

  EXPECT_TRUE(this->equals(iter1->key, key1));

  typename Database<string, value_type>::const_iterator iter2 = iter1++;

  EXPECT_FALSE(database->end() == iter1);
  EXPECT_FALSE(database->end() == iter2);

  EXPECT_TRUE(this->equals(iter2->key, key1));
  EXPECT_TRUE(this->equals(iter1->key, key2));

  typename Database<string, value_type>::const_iterator iter3 = iter1++;

  EXPECT_FALSE(database->end() == iter3);
  EXPECT_TRUE(this->equals(iter3->key, key2));
  EXPECT_TRUE(database->end() == iter1);

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewPutPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(database->put(key, value));

  EXPECT_TRUE(database->commit());

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewCommitPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));

  EXPECT_TRUE(database->commit());

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewGetPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(database->put(key, value));

  EXPECT_TRUE(database->commit());

  value_type new_value;

  EXPECT_TRUE(database->get(key, &new_value));

  EXPECT_TRUE(this->equals(value, new_value));

  database->close();
}

TYPED_TEST(DatabaseTest, TestNewGetNoCommitFails) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(database->put(key, value));

  value_type new_value;

  EXPECT_FALSE(database->get(key, &new_value));
}


TYPED_TEST(DatabaseTest, TestReadWritePutPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::ReadWrite));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(database->put(key, value));

  EXPECT_TRUE(database->commit());

  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteCommitPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::ReadWrite));

  EXPECT_TRUE(database->commit());

  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteGetPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(database->put(key, value));

  EXPECT_TRUE(database->commit());

  value_type new_value;

  EXPECT_TRUE(database->get(key, &new_value));

  EXPECT_TRUE(this->equals(value, new_value));

  database->close();
}

TYPED_TEST(DatabaseTest, TestReadWriteGetNoCommitFails) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(database->put(key, value));

  value_type new_value;

  EXPECT_FALSE(database->get(key, &new_value));
}

TYPED_TEST(DatabaseTest, TestReadOnlyPutFails) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));
  database->close();

  EXPECT_TRUE(database->open(name, Database<string, value_type>::ReadOnly));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_FALSE(database->put(key, value));
}

TYPED_TEST(DatabaseTest, TestReadOnlyCommitFails) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));
  database->close();

  EXPECT_TRUE(database->open(name, Database<string, value_type>::ReadOnly));

  EXPECT_FALSE(database->commit());
}

TYPED_TEST(DatabaseTest, TestReadOnlyGetPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(database->put(key, value));

  EXPECT_TRUE(database->commit());

  database->close();

  EXPECT_TRUE(database->open(name, Database<string, value_type>::ReadOnly));

  value_type new_value;

  EXPECT_TRUE(database->get(key, &new_value));

  EXPECT_TRUE(this->equals(value, new_value));
}

TYPED_TEST(DatabaseTest, TestReadOnlyGetNoCommitFails) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Database<string, value_type> > database =
      DatabaseFactory<string, value_type>(backend);
  EXPECT_TRUE(database->open(name, Database<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(database->put(key, value));

  database->close();

  EXPECT_TRUE(database->open(name, Database<string, value_type>::ReadOnly));

  value_type new_value;

  EXPECT_FALSE(database->get(key, &new_value));
}

#undef UNPACK_TYPES

}  // namespace caffe
