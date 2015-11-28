#include <string>
#include <vector>

#include "caffe/util/io.hpp"

#include "gtest/gtest.h"

#include "caffe/dataset_factory.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

namespace DatasetTest_internal {

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

}  // namespace DatasetTest_internal

#define UNPACK_TYPES \
  typedef typename TypeParam::value_type value_type; \
  const DataParameter_DB backend = TypeParam::backend;

template <typename TypeParam>
class DatasetTest : public ::testing::Test {
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
    return DatasetTest_internal::TestData<value_type>::TestValue();
  }

  string TestAltKey() {
    return "foo";
  }

  value_type TestAltValue() {
    return DatasetTest_internal::TestData<value_type>::TestAltValue();
  }

  template <typename T>
  bool equals(const T& a, const T& b) {
    return DatasetTest_internal::TestData<T>::equals(a, b);
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
const DataParameter_DB StringLmdb::backend = DataParameter_DB_LMDB;

struct VectorLeveldb {
  typedef vector<char> value_type;
  static const DataParameter_DB backend;
};
const DataParameter_DB VectorLeveldb::backend = DataParameter_DB_LEVELDB;

struct VectorLmdb {
  typedef vector<char> value_type;
  static const DataParameter_DB backend;
};
const DataParameter_DB VectorLmdb::backend = DataParameter_DB_LMDB;

struct DatumLeveldb {
  typedef Datum value_type;
  static const DataParameter_DB backend;
};
const DataParameter_DB DatumLeveldb::backend = DataParameter_DB_LEVELDB;

struct DatumLmdb {
  typedef Datum value_type;
  static const DataParameter_DB backend;
};
const DataParameter_DB DatumLmdb::backend = DataParameter_DB_LMDB;

typedef ::testing::Types<StringLeveldb, StringLmdb, VectorLeveldb, VectorLmdb,
    DatumLeveldb, DatumLmdb> TestTypes;

TYPED_TEST_CASE(DatasetTest, TestTypes);

TYPED_TEST(DatasetTest, TestNewDoesntExistPasses) {
  UNPACK_TYPES;

  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(this->DBName(),
      Dataset<string, value_type>::New));
  dataset->close();
}

TYPED_TEST(DatasetTest, TestNewExistsFails) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));
  dataset->close();

  EXPECT_FALSE(dataset->open(name, Dataset<string, value_type>::New));
}

TYPED_TEST(DatasetTest, TestReadOnlyExistsPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));
  dataset->close();

  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::ReadOnly));
  dataset->close();
}

TYPED_TEST(DatasetTest, TestReadOnlyDoesntExistFails) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_FALSE(dataset->open(name, Dataset<string, value_type>::ReadOnly));
}

TYPED_TEST(DatasetTest, TestReadWriteExistsPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));
  dataset->close();

  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::ReadWrite));
  dataset->close();
}

TYPED_TEST(DatasetTest, TestReadWriteDoesntExistPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::ReadWrite));
  dataset->close();
}

TYPED_TEST(DatasetTest, TestKeys) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  string key1 = this->TestKey();
  value_type value1 = this->TestValue();

  EXPECT_TRUE(dataset->put(key1, value1));

  string key2 = this->TestAltKey();
  value_type value2 = this->TestAltValue();

  EXPECT_TRUE(dataset->put(key2, value2));

  EXPECT_TRUE(dataset->commit());

  vector<string> keys;
  dataset->keys(&keys);

  EXPECT_EQ(2, keys.size());

  EXPECT_TRUE(this->equals(keys.at(0), key1) ||
      this->equals(keys.at(0), key2));
  EXPECT_TRUE(this->equals(keys.at(1), key1) ||
      this->equals(keys.at(2), key2));
  EXPECT_FALSE(this->equals(keys.at(0), keys.at(1)));
}

TYPED_TEST(DatasetTest, TestFirstKey) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  value_type value = this->TestValue();

  string key1 = "01";
  EXPECT_TRUE(dataset->put(key1, value));

  string key2 = "02";
  EXPECT_TRUE(dataset->put(key2, value));

  string key3 = "03";
  EXPECT_TRUE(dataset->put(key3, value));

  EXPECT_TRUE(dataset->commit());

  string first_key;
  dataset->first_key(&first_key);

  EXPECT_TRUE(this->equals(first_key, key1));
}

TYPED_TEST(DatasetTest, TestLastKey) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  value_type value = this->TestValue();

  string key1 = "01";
  EXPECT_TRUE(dataset->put(key1, value));

  string key2 = "02";
  EXPECT_TRUE(dataset->put(key2, value));

  string key3 = "03";
  EXPECT_TRUE(dataset->put(key3, value));

  EXPECT_TRUE(dataset->commit());

  string last_key;
  dataset->last_key(&last_key);

  EXPECT_TRUE(this->equals(last_key, key3));
}

TYPED_TEST(DatasetTest, TestFirstLastKeys) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  value_type value = this->TestValue();

  string key1 = "01";
  EXPECT_TRUE(dataset->put(key1, value));

  string key2 = "02";
  EXPECT_TRUE(dataset->put(key2, value));

  string key3 = "03";
  EXPECT_TRUE(dataset->put(key3, value));

  EXPECT_TRUE(dataset->commit());

  string first_key;
  dataset->first_key(&first_key);
  string last_key;
  dataset->last_key(&last_key);

  EXPECT_TRUE(this->equals(first_key, key1));
  EXPECT_TRUE(this->equals(last_key, key3));
}

TYPED_TEST(DatasetTest, TestFirstLastKeysUnOrdered) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  value_type value = this->TestValue();

  string key3 = "03";
  EXPECT_TRUE(dataset->put(key3, value));

  string key1 = "01";
  EXPECT_TRUE(dataset->put(key1, value));

  string key2 = "02";
  EXPECT_TRUE(dataset->put(key2, value));

  EXPECT_TRUE(dataset->commit());

  string first_key;
  dataset->first_key(&first_key);
  string last_key;
  dataset->last_key(&last_key);

  EXPECT_TRUE(this->equals(first_key, key1));
  EXPECT_TRUE(this->equals(last_key, key3));
}

TYPED_TEST(DatasetTest, TestKeysNoCommit) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  string key1 = this->TestKey();
  value_type value1 = this->TestValue();

  EXPECT_TRUE(dataset->put(key1, value1));

  string key2 = this->TestAltKey();
  value_type value2 = this->TestAltValue();

  EXPECT_TRUE(dataset->put(key2, value2));

  vector<string> keys;
  dataset->keys(&keys);

  EXPECT_EQ(0, keys.size());
}

TYPED_TEST(DatasetTest, TestIterators) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  const int kNumExamples = 4;
  for (int i = 0; i < kNumExamples; ++i) {
    stringstream ss;
    ss << i;
    string key = ss.str();
    ss << " here be data";
    value_type value = this->TestValue();
    EXPECT_TRUE(dataset->put(key, value));
  }
  EXPECT_TRUE(dataset->commit());

  int count = 0;
  typedef typename Dataset<string, value_type>::const_iterator Iter;
  for (Iter iter = dataset->begin(); iter != dataset->end(); ++iter) {
    (void)iter;
    ++count;
  }

  EXPECT_EQ(kNumExamples, count);
}

TYPED_TEST(DatasetTest, TestIteratorsPreIncrement) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  string key1 = this->TestAltKey();
  value_type value1 = this->TestAltValue();

  string key2 = this->TestKey();
  value_type value2 = this->TestValue();

  EXPECT_TRUE(dataset->put(key1, value1));
  EXPECT_TRUE(dataset->put(key2, value2));
  EXPECT_TRUE(dataset->commit());

  typename Dataset<string, value_type>::const_iterator iter1 =
      dataset->begin();

  EXPECT_FALSE(dataset->end() == iter1);

  EXPECT_TRUE(this->equals(iter1->key, key1));

  typename Dataset<string, value_type>::const_iterator iter2 = ++iter1;

  EXPECT_FALSE(dataset->end() == iter1);
  EXPECT_FALSE(dataset->end() == iter2);

  EXPECT_TRUE(this->equals(iter2->key, key2));

  typename Dataset<string, value_type>::const_iterator iter3 = ++iter2;

  EXPECT_TRUE(dataset->end() == iter3);

  dataset->close();
}

TYPED_TEST(DatasetTest, TestIteratorsPostIncrement) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  string key1 = this->TestAltKey();
  value_type value1 = this->TestAltValue();

  string key2 = this->TestKey();
  value_type value2 = this->TestValue();

  EXPECT_TRUE(dataset->put(key1, value1));
  EXPECT_TRUE(dataset->put(key2, value2));
  EXPECT_TRUE(dataset->commit());

  typename Dataset<string, value_type>::const_iterator iter1 =
      dataset->begin();

  EXPECT_FALSE(dataset->end() == iter1);

  EXPECT_TRUE(this->equals(iter1->key, key1));

  typename Dataset<string, value_type>::const_iterator iter2 = iter1++;

  EXPECT_FALSE(dataset->end() == iter1);
  EXPECT_FALSE(dataset->end() == iter2);

  EXPECT_TRUE(this->equals(iter2->key, key1));
  EXPECT_TRUE(this->equals(iter1->key, key2));

  typename Dataset<string, value_type>::const_iterator iter3 = iter1++;

  EXPECT_FALSE(dataset->end() == iter3);
  EXPECT_TRUE(this->equals(iter3->key, key2));
  EXPECT_TRUE(dataset->end() == iter1);

  dataset->close();
}

TYPED_TEST(DatasetTest, TestNewPutPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(dataset->put(key, value));

  EXPECT_TRUE(dataset->commit());

  dataset->close();
}

TYPED_TEST(DatasetTest, TestNewCommitPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  EXPECT_TRUE(dataset->commit());

  dataset->close();
}

TYPED_TEST(DatasetTest, TestNewGetPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(dataset->put(key, value));

  EXPECT_TRUE(dataset->commit());

  value_type new_value;

  EXPECT_TRUE(dataset->get(key, &new_value));

  EXPECT_TRUE(this->equals(value, new_value));

  dataset->close();
}

TYPED_TEST(DatasetTest, TestNewGetNoCommitFails) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(dataset->put(key, value));

  value_type new_value;

  EXPECT_FALSE(dataset->get(key, &new_value));
}


TYPED_TEST(DatasetTest, TestReadWritePutPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::ReadWrite));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(dataset->put(key, value));

  EXPECT_TRUE(dataset->commit());

  dataset->close();
}

TYPED_TEST(DatasetTest, TestReadWriteCommitPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::ReadWrite));

  EXPECT_TRUE(dataset->commit());

  dataset->close();
}

TYPED_TEST(DatasetTest, TestReadWriteGetPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(dataset->put(key, value));

  EXPECT_TRUE(dataset->commit());

  value_type new_value;

  EXPECT_TRUE(dataset->get(key, &new_value));

  EXPECT_TRUE(this->equals(value, new_value));

  dataset->close();
}

TYPED_TEST(DatasetTest, TestReadWriteGetNoCommitFails) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(dataset->put(key, value));

  value_type new_value;

  EXPECT_FALSE(dataset->get(key, &new_value));
}

TYPED_TEST(DatasetTest, TestReadOnlyPutFails) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));
  dataset->close();

  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::ReadOnly));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_FALSE(dataset->put(key, value));
}

TYPED_TEST(DatasetTest, TestReadOnlyCommitFails) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));
  dataset->close();

  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::ReadOnly));

  EXPECT_FALSE(dataset->commit());
}

TYPED_TEST(DatasetTest, TestReadOnlyGetPasses) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(dataset->put(key, value));

  EXPECT_TRUE(dataset->commit());

  dataset->close();

  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::ReadOnly));

  value_type new_value;

  EXPECT_TRUE(dataset->get(key, &new_value));

  EXPECT_TRUE(this->equals(value, new_value));
}

TYPED_TEST(DatasetTest, TestReadOnlyGetNoCommitFails) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();

  EXPECT_TRUE(dataset->put(key, value));

  dataset->close();

  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::ReadOnly));

  value_type new_value;

  EXPECT_FALSE(dataset->get(key, &new_value));
}

TYPED_TEST(DatasetTest, TestCreateManyItersShortScope) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();
  EXPECT_TRUE(dataset->put(key, value));
  EXPECT_TRUE(dataset->commit());

  for (int i = 0; i < 1000; ++i) {
    typename Dataset<string, value_type>::const_iterator iter =
        dataset->begin();
  }
}

TYPED_TEST(DatasetTest, TestCreateManyItersLongScope) {
  UNPACK_TYPES;

  string name = this->DBName();
  shared_ptr<Dataset<string, value_type> > dataset =
      DatasetFactory<string, value_type>(backend);
  EXPECT_TRUE(dataset->open(name, Dataset<string, value_type>::New));

  string key = this->TestKey();
  value_type value = this->TestValue();
  EXPECT_TRUE(dataset->put(key, value));
  EXPECT_TRUE(dataset->commit());

  vector<typename Dataset<string, value_type>::const_iterator> iters;
  for (int i = 0; i < 1000; ++i) {
    iters.push_back(dataset->begin());
  }
}

#undef UNPACK_TYPES

}  // namespace caffe
