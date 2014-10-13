#include <sys/stat.h>

#include <string>
#include <utility>

#include "caffe/lmdb_database.hpp"

namespace caffe {

bool LmdbDatabase::open(const string& filename, Mode mode) {
  LOG(INFO) << "LMDB: Open " << filename;

  CHECK(NULL == env_);
  CHECK(NULL == txn_);
  CHECK_EQ(0, dbi_);

  int retval;
  if (mode != ReadOnly) {
    retval = mkdir(filename.c_str(), 0744);
    switch (mode) {
    case New:
      if (0 != retval) {
        LOG(ERROR) << "mkdir " << filename << " failed";
        return false;
      }
      break;
    case ReadWrite:
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
  if (mode == ReadOnly) {
    flag1 = MDB_RDONLY | MDB_NOTLS;
    flag2 = MDB_RDONLY;
  }

  retval = mdb_env_open(env_, filename.c_str(), flag1, 0664);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_env_open failed " << mdb_strerror(retval);
    return false;
  }

  retval = mdb_txn_begin(env_, NULL, flag2, &txn_);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_txn_begin failed " << mdb_strerror(retval);
    return false;
  }

  retval = mdb_open(txn_, NULL, 0, &dbi_);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_open failed" << mdb_strerror(retval);
    return false;
  }

  return true;
}

bool LmdbDatabase::put(const key_type& key, const value_type& value) {
  LOG(INFO) << "LMDB: Put";

  // MDB_val::mv_size is not const, so we need to make a local copy.
  key_type local_key = key;
  value_type local_value = value;

  MDB_val mdbkey, mdbdata;
  mdbdata.mv_size = local_value.size();
  mdbdata.mv_data = local_value.data();
  mdbkey.mv_size = local_key.size();
  mdbkey.mv_data = local_key.data();

  CHECK_NOTNULL(txn_);
  CHECK_NE(0, dbi_);

  int retval = mdb_put(txn_, dbi_, &mdbkey, &mdbdata, 0);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_put failed " << mdb_strerror(retval);
    return false;
  }

  return true;
}

bool LmdbDatabase::get(const key_type& key, value_type* value) {
  LOG(INFO) << "LMDB: Get";

  key_type local_key = key;

  MDB_val mdbkey, mdbdata;
  mdbkey.mv_data = local_key.data();
  mdbkey.mv_size = local_key.size();

  int retval;
  MDB_txn* get_txn;
  retval = mdb_txn_begin(env_, NULL, MDB_RDONLY, &get_txn);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_txn_begin failed " << mdb_strerror(retval);
    return false;
  }

  retval = mdb_get(get_txn, dbi_, &mdbkey, &mdbdata);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_get failed " << mdb_strerror(retval);
    return false;
  }

  mdb_txn_abort(get_txn);

  Database::value_type temp_value(reinterpret_cast<char*>(mdbdata.mv_data),
      reinterpret_cast<char*>(mdbdata.mv_data) + mdbdata.mv_size);

  value->swap(temp_value);

  return true;
}

bool LmdbDatabase::commit() {
  LOG(INFO) << "LMDB: Commit";

  CHECK_NOTNULL(txn_);

  int retval;
  retval = mdb_txn_commit(txn_);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_txn_commit failed " << mdb_strerror(retval);
    return false;
  }

  retval = mdb_txn_begin(env_, NULL, 0, &txn_);
  if (MDB_SUCCESS != retval) {
    LOG(ERROR) << "mdb_txn_begin failed " << mdb_strerror(retval);
    return false;
  }

  return true;
}

void LmdbDatabase::close() {
  LOG(INFO) << "LMDB: Close";

  if (env_ && dbi_) {
    mdb_close(env_, dbi_);
    mdb_env_close(env_);
    env_ = NULL;
    dbi_ = 0;
    txn_ = NULL;
  }
}

LmdbDatabase::const_iterator LmdbDatabase::begin() const {
  MDB_cursor* cursor;
  int retval;
  retval = mdb_cursor_open(txn_, dbi_, &cursor);
  CHECK_EQ(retval, MDB_SUCCESS) << mdb_strerror(retval);
  MDB_val key;
  MDB_val val;
  retval = mdb_cursor_get(cursor, &key, &val, MDB_FIRST);
  CHECK_EQ(retval, MDB_SUCCESS) << mdb_strerror(retval);

  shared_ptr<DatabaseState> state(new LmdbState(cursor, txn_, &dbi_));
  return const_iterator(this, state);
}

LmdbDatabase::const_iterator LmdbDatabase::end() const {
  shared_ptr<DatabaseState> state(new LmdbState(NULL, txn_, &dbi_));
  return const_iterator(this, state);
}

LmdbDatabase::const_iterator LmdbDatabase::cbegin() const { return begin(); }
LmdbDatabase::const_iterator LmdbDatabase::cend() const { return end(); }

bool LmdbDatabase::equal(shared_ptr<DatabaseState> state1,
    shared_ptr<DatabaseState> state2) const {
  shared_ptr<LmdbState> lmdb_state1 =
      boost::dynamic_pointer_cast<LmdbState>(state1);

  CHECK_NOTNULL(lmdb_state1.get());

  shared_ptr<LmdbState> lmdb_state2 =
      boost::dynamic_pointer_cast<LmdbState>(state2);

  CHECK_NOTNULL(lmdb_state2.get());

  // The KV store doesn't really have any sort of ordering,
  // so while we can do a sequential scan over the collection,
  // we can't really use subranges.
  return !lmdb_state1->cursor_ && !lmdb_state2->cursor_;
}

void LmdbDatabase::increment(shared_ptr<DatabaseState> state) const {
  shared_ptr<LmdbState> lmdb_state =
      boost::dynamic_pointer_cast<LmdbState>(state);

  CHECK_NOTNULL(lmdb_state.get());

  MDB_cursor*& cursor = lmdb_state->cursor_;

  CHECK_NOTNULL(cursor);

  MDB_val key;
  MDB_val val;
  int retval = mdb_cursor_get(cursor, &key, &val, MDB_NEXT);
  if (MDB_NOTFOUND == retval) {
    mdb_cursor_close(cursor);
    cursor = NULL;
  } else {
    CHECK_EQ(MDB_SUCCESS, retval) << mdb_strerror(retval);
  }
}

Database::KV& LmdbDatabase::dereference(shared_ptr<DatabaseState> state) const {
  shared_ptr<LmdbState> lmdb_state =
      boost::dynamic_pointer_cast<LmdbState>(state);

  CHECK_NOTNULL(lmdb_state.get());

  MDB_cursor*& cursor = lmdb_state->cursor_;

  CHECK_NOTNULL(cursor);

  MDB_val mdb_key;
  MDB_val mdb_val;
  int retval = mdb_cursor_get(cursor, &mdb_key, &mdb_val, MDB_GET_CURRENT);
  CHECK_EQ(retval, MDB_SUCCESS) << mdb_strerror(retval);

  char* key_data = reinterpret_cast<char*>(mdb_key.mv_data);
  char* value_data = reinterpret_cast<char*>(mdb_val.mv_data);

  Database::key_type temp_key(key_data, key_data + mdb_key.mv_size);

  Database::value_type temp_value(value_data,
      value_data + mdb_val.mv_size);

  lmdb_state->kv_pair_.key.swap(temp_key);
  lmdb_state->kv_pair_.value.swap(temp_value);

  return lmdb_state->kv_pair_;
}

}  // namespace caffe
