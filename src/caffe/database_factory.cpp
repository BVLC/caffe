#include <string>
#include <utility>

#include "caffe/database_factory.hpp"
#include "caffe/leveldb_database.hpp"
#include "caffe/lmdb_database.hpp"

namespace caffe {

shared_ptr<Database> DatabaseFactory(const DataParameter_DB& type) {
  switch (type) {
  case DataParameter_DB_LEVELDB:
    return shared_ptr<Database>(new LeveldbDatabase());
  case DataParameter_DB_LMDB:
    return shared_ptr<Database>(new LmdbDatabase());
  default:
    LOG(FATAL) << "Unknown database type " << type;
    return shared_ptr<Database>();
  }
}

shared_ptr<Database> DatabaseFactory(const string& type) {
  if ("leveldb" == type) {
    return DatabaseFactory(DataParameter_DB_LEVELDB);
  } else if ("lmdb" == type) {
    return DatabaseFactory(DataParameter_DB_LMDB);
  } else {
    LOG(FATAL) << "Unknown database type " << type;
    return shared_ptr<Database>();
  }
}

}  // namespace caffe


