#ifndef CAFFE_LEVELDB_DATASET_H_
#define CAFFE_LEVELDB_DATASET_H_

#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/dataset.hpp"

namespace caffe {

template <typename K, typename V>
class LeveldbDataset : public Dataset<K, V> {
 public:
  typedef Dataset<K, V> Base;
  typedef typename Base::key_type key_type;
  typedef typename Base::value_type value_type;
  typedef typename Base::DatasetState DatasetState;
  typedef typename Base::Mode Mode;
  typedef typename Base::const_iterator const_iterator;
  typedef typename Base::KV KV;

  bool open(const string& filename, Mode mode);
  bool put(const K& key, const V& value);
  bool get(const K& key, V* value);
  bool commit();
  void close();

  void keys(vector<K>* keys);

  const_iterator begin() const;
  const_iterator cbegin() const;
  const_iterator end() const;
  const_iterator cend() const;

 protected:
  class LeveldbState : public DatasetState {
   public:
    explicit LeveldbState(shared_ptr<leveldb::DB> db,
        shared_ptr<leveldb::Iterator> iter)
        : DatasetState(),
          db_(db),
          iter_(iter) { }

    ~LeveldbState() {
      // This order is very important.
      // Iterators must be destroyed before their associated DB
      // is destroyed.
      iter_.reset();
      db_.reset();
    }

    shared_ptr<DatasetState> clone() {
      shared_ptr<leveldb::Iterator> new_iter;

      if (iter_.get()) {
        new_iter.reset(db_->NewIterator(leveldb::ReadOptions()));
        CHECK(iter_->Valid());
        new_iter->Seek(iter_->key());
        CHECK(new_iter->Valid());
      }

      return shared_ptr<DatasetState>(new LeveldbState(db_, new_iter));
    }

    shared_ptr<leveldb::DB> db_;
    shared_ptr<leveldb::Iterator> iter_;
    KV kv_pair_;
  };

  bool equal(shared_ptr<DatasetState> state1,
      shared_ptr<DatasetState> state2) const;
  void increment(shared_ptr<DatasetState>* state) const;
  KV& dereference(shared_ptr<DatasetState> state) const;

  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::WriteBatch> batch_;
  bool read_only_;
};

}  // namespace caffe

#endif  // CAFFE_LEVELDB_DATASET_H_
