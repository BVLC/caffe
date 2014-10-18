#ifndef CAFFE_LMDB_DATASET_H_
#define CAFFE_LMDB_DATASET_H_

#include <string>
#include <utility>
#include <vector>

#include "lmdb.h"

#include "caffe/common.hpp"
#include "caffe/dataset.hpp"

namespace caffe {

template <typename K, typename V,
          typename KCoder = dataset_internal::DefaultCoder<K>,
          typename VCoder = dataset_internal::DefaultCoder<V> >
class LmdbDataset : public Dataset<K, V, KCoder, VCoder> {
 public:
  typedef Dataset<K, V, KCoder, VCoder> Base;
  typedef typename Base::key_type key_type;
  typedef typename Base::value_type value_type;
  typedef typename Base::DatasetState DatasetState;
  typedef typename Base::Mode Mode;
  typedef typename Base::const_iterator const_iterator;
  typedef typename Base::KV KV;

  LmdbDataset()
      : env_(NULL),
        dbi_(0),
        write_txn_(NULL),
        read_txn_(NULL) { }

  bool open(const string& filename, Mode mode);
  bool put(const K& key, const V& value);
  bool get(const K& key, V* value);
  bool first_key(K* key);
  bool last_key(K* key);
  bool commit();
  void close();

  void keys(vector<K>* keys);

  const_iterator begin() const;
  const_iterator cbegin() const;
  const_iterator end() const;
  const_iterator cend() const;

 protected:
  class LmdbState : public DatasetState {
   public:
    explicit LmdbState(MDB_cursor* cursor, MDB_txn* txn, const MDB_dbi* dbi)
        : DatasetState(),
          cursor_(cursor),
          txn_(txn),
          dbi_(dbi) { }

    shared_ptr<DatasetState> clone() {
      CHECK(cursor_);

      MDB_cursor* new_cursor;
      int retval;

      retval = mdb_cursor_open(txn_, *dbi_, &new_cursor);
      CHECK_EQ(retval, MDB_SUCCESS) << mdb_strerror(retval);
      MDB_val key;
      MDB_val val;
      retval = mdb_cursor_get(cursor_, &key, &val, MDB_GET_CURRENT);
      CHECK_EQ(retval, MDB_SUCCESS) << mdb_strerror(retval);
      retval = mdb_cursor_get(new_cursor, &key, &val, MDB_SET);
      CHECK_EQ(MDB_SUCCESS, retval) << mdb_strerror(retval);

      return shared_ptr<DatasetState>(new LmdbState(new_cursor, txn_, dbi_));
    }

    MDB_cursor* cursor_;
    MDB_txn* txn_;
    const MDB_dbi* dbi_;
    KV kv_pair_;
  };

  bool equal(shared_ptr<DatasetState> state1,
      shared_ptr<DatasetState> state2) const;
  void increment(shared_ptr<DatasetState>* state) const;
  KV& dereference(shared_ptr<DatasetState> state) const;

  MDB_env* env_;
  MDB_dbi dbi_;
  MDB_txn* write_txn_;
  MDB_txn* read_txn_;
};

}  // namespace caffe

#endif  // CAFFE_LMDB_DATASET_H_
