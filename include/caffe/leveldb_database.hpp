#ifndef CAFFE_LEVELDB_DATABASE_H_
#define CAFFE_LEVELDB_DATABASE_H_

#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <string>
#include <utility>

#include "caffe/common.hpp"
#include "caffe/database.hpp"

namespace caffe {

class LeveldbDatabase : public Database {
 public:
  void open(const string& filename, Mode mode);
  void put(buffer_t* key, buffer_t* value);
  void get(buffer_t* key, buffer_t* value);
  void commit();
  void close();

  const_iterator begin() const;
  const_iterator cbegin() const;
  const_iterator end() const;
  const_iterator cend() const;

 protected:
  class LeveldbState : public Database::DatabaseState {
   public:
    explicit LeveldbState(shared_ptr<leveldb::Iterator> iter)
        : Database::DatabaseState(),
          iter_(iter) { }

    shared_ptr<leveldb::Iterator> iter_;
    KV kv_pair_;
  };

  bool equal(shared_ptr<DatabaseState> state1,
      shared_ptr<DatabaseState> state2) const;
  void increment(shared_ptr<DatabaseState> state) const;
  Database::KV& dereference(shared_ptr<DatabaseState> state) const;

  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::WriteBatch> batch_;
  bool read_only_;
};

}  // namespace caffe

#endif  // CAFFE_LEVELDB_DATABASE_H_
