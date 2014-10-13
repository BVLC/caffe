#ifndef CAFFE_LEVELDB_DATABASE_H_
#define CAFFE_LEVELDB_DATABASE_H_

#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/database.hpp"

namespace caffe {

class LeveldbDatabase : public Database {
 public:
  bool open(const string& filename, Mode mode);
  bool put(const key_type& key, const value_type& value);
  bool get(const key_type& key, value_type* value);
  bool commit();
  void close();

  void keys(vector<key_type>* keys);

  const_iterator begin() const;
  const_iterator cbegin() const;
  const_iterator end() const;
  const_iterator cend() const;

 protected:
  class LeveldbState : public Database::DatabaseState {
   public:
    explicit LeveldbState(shared_ptr<leveldb::DB> db,
        shared_ptr<leveldb::Iterator> iter)
        : Database::DatabaseState(),
          db_(db),
          iter_(iter) { }

    ~LeveldbState() {
      // This order is very important.
      // Iterators must be destroyed before their associated DB
      // is destroyed.
      iter_.reset();
      db_.reset();
    }

    shared_ptr<DatabaseState> clone() {
      shared_ptr<leveldb::Iterator> new_iter;

      if (iter_.get()) {
        new_iter.reset(db_->NewIterator(leveldb::ReadOptions()));
        CHECK(iter_->Valid());
        new_iter->Seek(iter_->key());
        CHECK(new_iter->Valid());
      }

      return shared_ptr<DatabaseState>(new LeveldbState(db_, new_iter));
    }

    shared_ptr<leveldb::DB> db_;
    shared_ptr<leveldb::Iterator> iter_;
    KV kv_pair_;
  };

  bool equal(shared_ptr<DatabaseState> state1,
      shared_ptr<DatabaseState> state2) const;
  void increment(shared_ptr<DatabaseState>* state) const;
  Database::KV& dereference(shared_ptr<DatabaseState> state) const;

  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::WriteBatch> batch_;
  bool read_only_;
};

}  // namespace caffe

#endif  // CAFFE_LEVELDB_DATABASE_H_
