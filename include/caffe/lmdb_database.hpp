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
    explicit LmdbState(MDB_cursor* cursor, MDB_txn* txn, const MDB_dbi* dbi)
        : Database::DatabaseState(),
          cursor_(cursor),
          txn_(txn),
          dbi_(dbi) { }

    shared_ptr<DatabaseState> clone() {
      MDB_cursor* new_cursor;

      if (cursor_) {
        int retval;
        retval = mdb_cursor_open(txn_, *dbi_, &new_cursor);
        CHECK_EQ(retval, MDB_SUCCESS) << mdb_strerror(retval);
        MDB_val key;
        MDB_val val;
        retval = mdb_cursor_get(cursor_, &key, &val, MDB_GET_CURRENT);
        CHECK_EQ(retval, MDB_SUCCESS) << mdb_strerror(retval);
        retval = mdb_cursor_get(new_cursor, &key, &val, MDB_SET);
        CHECK_EQ(MDB_SUCCESS, retval) << mdb_strerror(retval);
      } else {
        new_cursor = cursor_;
      }

      return shared_ptr<DatabaseState>(new LmdbState(new_cursor, txn_, dbi_));
    }

    MDB_cursor* cursor_;
    MDB_txn* txn_;
    const MDB_dbi* dbi_;
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
