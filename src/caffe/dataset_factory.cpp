#include <string>
#include <utility>
#include <vector>

#include "caffe/dataset_factory.hpp"
#include "caffe/leveldb_dataset.hpp"
#include "caffe/lmdb_dataset.hpp"

namespace caffe {

template <typename K, typename V>
shared_ptr<Dataset<K, V> > DatasetFactory(const DataParameter_DB& type) {
  switch (type) {
  case DataParameter_DB_LEVELDB:
    return shared_ptr<Dataset<K, V> >(new LeveldbDataset<K, V>());
  case DataParameter_DB_LMDB:
    return shared_ptr<Dataset<K, V> >(new LmdbDataset<K, V>());
  default:
    LOG(FATAL) << "Unknown dataset type " << type;
    return shared_ptr<Dataset<K, V> >();
  }
}

template <typename K, typename V>
shared_ptr<Dataset<K, V> > DatasetFactory(const string& type) {
  if ("leveldb" == type) {
    return DatasetFactory<K, V>(DataParameter_DB_LEVELDB);
  } else if ("lmdb" == type) {
    return DatasetFactory<K, V>(DataParameter_DB_LMDB);
  } else {
    LOG(FATAL) << "Unknown dataset type " << type;
    return shared_ptr<Dataset<K, V> >();
  }
}

#define REGISTER_DATASET(key_type, value_type) \
  template shared_ptr<Dataset<key_type, value_type> > \
      DatasetFactory(const string& type); \
  template shared_ptr<Dataset<key_type, value_type> > \
      DatasetFactory(const DataParameter_DB& type); \

REGISTER_DATASET(string, string);
REGISTER_DATASET(string, vector<char>);
REGISTER_DATASET(string, Datum);

#undef REGISTER_DATASET

}  // namespace caffe


