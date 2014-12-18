#include <sys/stat.h>

#include <string>
#include <utility>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/lmdb_dataset.hpp"

namespace caffe {

template <typename K, typename V, typename KCoder, typename VCoder>
bool LmdbDataset<K, V, KCoder, VCoder>::open(const string& filename,
    Mode mode) {
  DLOG(INFO) << "LMDB: Open " << filename;

  CHECK(NULL == env_);
  CHECK(NULL == write_txn_);
  CHECK(NULL == read_txn_);
  CHECK_EQ(0, dbi_);

  int retval;
  if (mode != Base::ReadOnly) {
    retval = mkdir(filename.c_str(), 0744);
    switch (mode) {
    case Base::New:
      if (0 != retval) {
        LOG(ERROR) << "mkdir " << filename << " failed";
        return false;
      }
      break;
    case Base::ReadWrite:
      if (-1 == retval && EEXIST != errno) {
        LOG(ERROR) << "mkdir " << filename << " failed ("
            << strerror(errno) << ")";
        return false;
      }
      break;
    default:
      LOG(FATAL) << "Invalid mode " << mode;
    }
  }

  retval = mdb_env_create(&env_);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_env_create failed "
        << mdb_strerror(retval);
    return false;
  }

  retval = mdb_env_set_mapsize(env_, 1099511627776);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_env_set_mapsize failed " << mdb_strerror(retval);
    return false;
  }

  int flag1 = 0;
  int flag2 = 0;
  if (mode == Base::ReadOnly) {
    flag1 = MDB_RDONLY | MDB_NOTLS;
    flag2 = MDB_RDONLY;
  }

  retval = mdb_env_open(env_, filename.c_str(), flag1, 0664);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_env_open failed " << mdb_strerror(retval);
    return false;
  }

  retval = mdb_txn_begin(env_, NULL, MDB_RDONLY, &read_txn_);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_txn_begin failed " << mdb_strerror(retval);
    return false;
  }

  retval = mdb_txn_begin(env_, NULL, flag2, &write_txn_);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_txn_begin failed " << mdb_strerror(retval);
    return false;
  }

  retval = mdb_open(write_txn_, NULL, 0, &dbi_);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_open failed" << mdb_strerror(retval);
    return false;
  }

  return true;
}

template <typename K, typename V, typename KCoder, typename VCoder>
bool LmdbDataset<K, V, KCoder, VCoder>::put(const K& key, const V& value) {
  DLOG(INFO) << "LMDB: Put";

  vector<char> serialized_key;
  if (!KCoder::serialize(key, &serialized_key)) {
    LOG(ERROR) << "failed to serialize key";
    return false;
  }

  vector<char> serialized_value;
  if (!VCoder::serialize(value, &serialized_value)) {
    LOG(ERROR) << "failed to serialized value";
    return false;
  }

  MDB_val mdbkey, mdbdata;
  mdbdata.mv_size = serialized_value.size();
  mdbdata.mv_data = serialized_value.data();
  mdbkey.mv_size = serialized_key.size();
  mdbkey.mv_data = serialized_key.data();

  CHECK_NOTNULL(write_txn_);
  CHECK_NE(0, dbi_);

  int retval = mdb_put(write_txn_, dbi_, &mdbkey, &mdbdata, 0);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_put failed " << mdb_strerror(retval);
    return false;
  }

  return true;
}

template <typename K, typename V, typename KCoder, typename VCoder>
bool LmdbDataset<K, V, KCoder, VCoder>::get(const K& key, V* value) {
  DLOG(INFO) << "LMDB: Get";

  vector<char> serialized_key;
  if (!KCoder::serialize(key, &serialized_key)) {
    LOG(ERROR) << "failed to serialized key";
    return false;
  }

  MDB_val mdbkey, mdbdata;
  mdbkey.mv_data = serialized_key.data();
  mdbkey.mv_size = serialized_key.size();

  int retval;
  retval = mdb_get(read_txn_, dbi_, &mdbkey, &mdbdata);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_get failed " << mdb_strerror(retval);
    return false;
  }

  if (!VCoder::deserialize(reinterpret_cast<char*>(mdbdata.mv_data),
      mdbdata.mv_size, value)) {
    LOG(ERROR) << "failed to deserialize value";
    return false;
  }

  return true;
}

template <typename K, typename V, typename KCoder, typename VCoder>
bool LmdbDataset<K, V, KCoder, VCoder>::first_key(K* key) {
  DLOG(INFO) << "LMDB: First key";

  int retval;

  MDB_cursor* cursor;
  retval = mdb_cursor_open(read_txn_, dbi_, &cursor);
  CHECK_EQ(retval, MDB_SUCCESS) << mdb_strerror(retval);
  MDB_val mdbkey;
  MDB_val mdbval;
  retval = mdb_cursor_get(cursor, &mdbkey, &mdbval, MDB_FIRST);
  CHECK_EQ(retval, MDB_SUCCESS) << mdb_strerror(retval);

  mdb_cursor_close(cursor);

  if (!KCoder::deserialize(reinterpret_cast<char*>(mdbkey.mv_data),
      mdbkey.mv_size, key)) {
    LOG(ERROR) << "failed to deserialize key";
    return false;
  }

  return true;
}

template <typename K, typename V, typename KCoder, typename VCoder>
bool LmdbDataset<K, V, KCoder, VCoder>::last_key(K* key) {
  DLOG(INFO) << "LMDB: Last key";

  int retval;

  MDB_cursor* cursor;
  retval = mdb_cursor_open(read_txn_, dbi_, &cursor);
  CHECK_EQ(retval, MDB_SUCCESS) << mdb_strerror(retval);
  MDB_val mdbkey;
  MDB_val mdbval;
  retval = mdb_cursor_get(cursor, &mdbkey, &mdbval, MDB_LAST);
  CHECK_EQ(retval, MDB_SUCCESS) << mdb_strerror(retval);

  mdb_cursor_close(cursor);

  if (!KCoder::deserialize(reinterpret_cast<char*>(mdbkey.mv_data),
      mdbkey.mv_size, key)) {
    LOG(ERROR) << "failed to deserialize key";
    return false;
  }

  return true;
}

template <typename K, typename V, typename KCoder, typename VCoder>
bool LmdbDataset<K, V, KCoder, VCoder>::commit() {
  DLOG(INFO) << "LMDB: Commit";

  CHECK_NOTNULL(write_txn_);

  int retval;
  retval = mdb_txn_commit(write_txn_);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_txn_commit failed " << mdb_strerror(retval);
    return false;
  }

  mdb_txn_abort(read_txn_);

  retval = mdb_txn_begin(env_, NULL, 0, &write_txn_);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_txn_begin failed " << mdb_strerror(retval);
    return false;
  }

  retval = mdb_txn_begin(env_, NULL, MDB_RDONLY, &read_txn_);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_txn_begin failed " << mdb_strerror(retval);
    return false;
  }

  return true;
}

template <typename K, typename V, typename KCoder, typename VCoder>
void LmdbDataset<K, V, KCoder, VCoder>::close() {
  DLOG(INFO) << "LMDB: Close";

  if (env_ && dbi_) {
    mdb_txn_abort(write_txn_);
    mdb_txn_abort(read_txn_);
    mdb_close(env_, dbi_);
    mdb_env_close(env_);
    env_ = NULL;
    dbi_ = 0;
    write_txn_ = NULL;
    read_txn_ = NULL;
  }
}

template <typename K, typename V, typename KCoder, typename VCoder>
void LmdbDataset<K, V, KCoder, VCoder>::keys(vector<K>* keys) {
  DLOG(INFO) << "LMDB: Keys";

  keys->clear();
  for (const_iterator iter = begin(); iter != end(); ++iter) {
    keys->push_back(iter->key);
  }
}

template <typename K, typename V, typename KCoder, typename VCoder>
typename LmdbDataset<K, V, KCoder, VCoder>::const_iterator
    LmdbDataset<K, V, KCoder, VCoder>::begin() const {
  int retval;

  MDB_cursor* cursor;
  retval = mdb_cursor_open(read_txn_, dbi_, &cursor);
  CHECK_EQ(retval, MDB_SUCCESS) << mdb_strerror(retval);
  MDB_val key;
  MDB_val val;
  retval = mdb_cursor_get(cursor, &key, &val, MDB_FIRST);

  CHECK(MDB_SUCCESS == retval || MDB_NOTFOUND == retval)
      << mdb_strerror(retval);

  shared_ptr<DatasetState> state;
  if (MDB_SUCCESS == retval) {
    state.reset(new LmdbState(cursor, read_txn_, &dbi_));
  } else {
    mdb_cursor_close(cursor);
  }
  return const_iterator(this, state);
}

template <typename K, typename V, typename KCoder, typename VCoder>
typename LmdbDataset<K, V, KCoder, VCoder>::const_iterator
    LmdbDataset<K, V, KCoder, VCoder>::end() const {
  shared_ptr<DatasetState> state;
  return const_iterator(this, state);
}

template <typename K, typename V, typename KCoder, typename VCoder>
typename LmdbDataset<K, V, KCoder, VCoder>::const_iterator
    LmdbDataset<K, V, KCoder, VCoder>::cbegin() const { return begin(); }

template <typename K, typename V, typename KCoder, typename VCoder>
typename LmdbDataset<K, V, KCoder, VCoder>::const_iterator
    LmdbDataset<K, V, KCoder, VCoder>::cend() const { return end(); }

template <typename K, typename V, typename KCoder, typename VCoder>
bool LmdbDataset<K, V, KCoder, VCoder>::equal(shared_ptr<DatasetState> state1,
    shared_ptr<DatasetState> state2) const {
  shared_ptr<LmdbState> lmdb_state1 =
      boost::dynamic_pointer_cast<LmdbState>(state1);

  shared_ptr<LmdbState> lmdb_state2 =
      boost::dynamic_pointer_cast<LmdbState>(state2);

  // The KV store doesn't really have any sort of ordering,
  // so while we can do a sequential scan over the collection,
  // we can't really use subranges.
  return !lmdb_state1 && !lmdb_state2;
}

template <typename K, typename V, typename KCoder, typename VCoder>
void LmdbDataset<K, V, KCoder, VCoder>::increment(
    shared_ptr<DatasetState>* state) const {
  shared_ptr<LmdbState> lmdb_state =
      boost::dynamic_pointer_cast<LmdbState>(*state);

  CHECK_NOTNULL(lmdb_state.get());

  MDB_cursor*& cursor = lmdb_state->cursor_;

  CHECK_NOTNULL(cursor);

  MDB_val key;
  MDB_val val;
  int retval = mdb_cursor_get(cursor, &key, &val, MDB_NEXT);
  if (MDB_NOTFOUND == retval) {
    mdb_cursor_close(cursor);
    state->reset();
  } else {
    CHECK_EQ(MDB_SUCCESS, retval) << mdb_strerror(retval);
  }
}

template <typename K, typename V, typename KCoder, typename VCoder>
typename Dataset<K, V, KCoder, VCoder>::KV&
    LmdbDataset<K, V, KCoder, VCoder>::dereference(
    shared_ptr<DatasetState> state) const {
  shared_ptr<LmdbState> lmdb_state =
      boost::dynamic_pointer_cast<LmdbState>(state);

  CHECK_NOTNULL(lmdb_state.get());

  MDB_cursor*& cursor = lmdb_state->cursor_;

  CHECK_NOTNULL(cursor);

  MDB_val mdb_key;
  MDB_val mdb_val;
  int retval = mdb_cursor_get(cursor, &mdb_key, &mdb_val, MDB_GET_CURRENT);
  CHECK_EQ(retval, MDB_SUCCESS) << mdb_strerror(retval);

  CHECK(KCoder::deserialize(reinterpret_cast<char*>(mdb_key.mv_data),
      mdb_key.mv_size, &lmdb_state->kv_pair_.key));
  CHECK(VCoder::deserialize(reinterpret_cast<char*>(mdb_val.mv_data),
      mdb_val.mv_size, &lmdb_state->kv_pair_.value));

  return lmdb_state->kv_pair_;
}

INSTANTIATE_DATASET(LmdbDataset);

}  // namespace caffe
