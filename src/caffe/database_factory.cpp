#include <string>
#include <utility>
#include <vector>

#include "caffe/database_factory.hpp"
#include "caffe/leveldb_database.hpp"
#include "caffe/lmdb_database.hpp"

namespace caffe {

template <typename K, typename V>
shared_ptr<Database<K, V> > DatabaseFactory(const DataParameter_DB& type) {
  switch (type) {
  case DataParameter_DB_LEVELDB:
    return shared_ptr<Database<K, V> >(new LeveldbDatabase<K, V>());
  case DataParameter_DB_LMDB:
    return shared_ptr<Database<K, V> >(new LmdbDatabase<K, V>());
  default:
    LOG(FATAL) << "Unknown database type " << type;
    return shared_ptr<Database<K, V> >();
  }
}

template <typename K, typename V>
shared_ptr<Database<K, V> > DatabaseFactory(const string& type) {
  if ("leveldb" == type) {
    return DatabaseFactory<K, V>(DataParameter_DB_LEVELDB);
  } else if ("lmdb" == type) {
    return DatabaseFactory<K, V>(DataParameter_DB_LMDB);
  } else {
    LOG(FATAL) << "Unknown database type " << type;
    return shared_ptr<Database<K, V> >();
  }
}

#define REGISTER_DATABASE(key_type, value_type) \
  template shared_ptr<Database<key_type, value_type> > \
      DatabaseFactory(const string& type); \
  template shared_ptr<Database<key_type, value_type> > \
      DatabaseFactory(const DataParameter_DB& type); \

REGISTER_DATABASE(string, string);
REGISTER_DATABASE(string, vector<char>);
REGISTER_DATABASE(string, Datum);

#undef REGISTER_DATABASE

}  // namespace caffe


