#ifndef CAFFE_LMDB_DATABASE_H_
#define CAFFE_LMDB_DATABASE_H_

#include <string>
#include <utility>

#include "lmdb.h"

#include "caffe/common.hpp"
#include "caffe/database.hpp"

namespace caffe {

class LmdbDatabase : public Database {
 public:
  LmdbDatabase()
      : env_(NULL),
        dbi_(0),
        txn_(NULL) { }

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
  class LmdbState : public Database::DatabaseState {
   public:
    explicit LmdbState(MDB_cursor* cursor)
        : Database::DatabaseState(),
          cursor_(cursor) { }

    MDB_cursor* cursor_;
    KV kv_pair_;
  };

  bool equal(shared_ptr<DatabaseState> state1,
      shared_ptr<DatabaseState> state2) const;
  void increment(shared_ptr<DatabaseState> state) const;
  Database::KV& dereference(shared_ptr<DatabaseState> state) const;

  MDB_env* env_;
  MDB_dbi dbi_;
  MDB_txn* txn_;
};

}  // namespace caffe

#endif  // CAFFE_LMDB_DATABASE_H_
