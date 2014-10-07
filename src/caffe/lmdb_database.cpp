#include <sys/stat.h>

#include <string>
#include <utility>

#include "caffe/lmdb_database.hpp"

namespace caffe {

void LmdbDatabase::open(const string& filename, Mode mode) {
  LOG(INFO) << "LMDB: Open " << filename;

  CHECK(NULL == env_);
  CHECK(NULL == txn_);
  CHECK_EQ(0, dbi_);

  if (mode != ReadOnly) {
    CHECK_EQ(mkdir(filename.c_str(), 0744), 0) << "mkdir " << filename
                                                << "failed";
  }

  CHECK_EQ(mdb_env_create(&env_), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(env_, 1099511627776), MDB_SUCCESS)  // 1TB
      << "mdb_env_set_mapsize failed";

  int flag1 = 0;
  int flag2 = 0;
  if (mode == ReadOnly) {
    flag1 = MDB_RDONLY | MDB_NOTLS;
    flag2 = MDB_RDONLY;
  }

  CHECK_EQ(mdb_env_open(env_, filename.c_str(), flag1, 0664), MDB_SUCCESS)
      << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(env_, NULL, flag2, &txn_), MDB_SUCCESS)
      << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(txn_, NULL, 0, &dbi_), MDB_SUCCESS) << "mdb_open failed";
}

void LmdbDatabase::put(const string& key, const string& value) {
  LOG(INFO) << "LMDB: Put " << key;

  MDB_val mdbkey, mdbdata;
  mdbdata.mv_size = value.size();
  mdbdata.mv_data = const_cast<char*>(&value[0]);
  mdbkey.mv_size = key.size();
  mdbkey.mv_data = const_cast<char*>(&key[0]);

  CHECK_NOTNULL(txn_);
  CHECK_NE(0, dbi_);

  CHECK_EQ(mdb_put(txn_, dbi_, &mdbkey, &mdbdata, 0), MDB_SUCCESS)
      << "mdb_put failed";
}

void LmdbDatabase::commit() {
  LOG(INFO) << "LMDB: Commit";

  CHECK_NOTNULL(txn_);

  CHECK_EQ(mdb_txn_commit(txn_), MDB_SUCCESS) << "mdb_txn_commit failed";
}

void LmdbDatabase::close() {
  LOG(INFO) << "LMDB: Close";

  if (env_ && dbi_ && txn_) {
    this->commit();
  }

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
  CHECK_EQ(mdb_cursor_open(txn_, dbi_, &cursor), MDB_SUCCESS);
  MDB_val key;
  MDB_val val;
  CHECK_EQ(mdb_cursor_get(cursor, &key, &val, MDB_FIRST), MDB_SUCCESS);

  shared_ptr<DatabaseState> state(new LmdbState(cursor));
  return const_iterator(this, state);
}

LmdbDatabase::const_iterator LmdbDatabase::end() const {
  shared_ptr<DatabaseState> state(new LmdbState(NULL));
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

  MDB_val key;
  MDB_val val;
  if (MDB_SUCCESS != mdb_cursor_get(cursor, &key, &val, MDB_NEXT)) {
    mdb_cursor_close(cursor);
    cursor = NULL;
  }
}

pair<string, string>& LmdbDatabase::dereference(
    shared_ptr<DatabaseState> state) const {
  shared_ptr<LmdbState> lmdb_state =
      boost::dynamic_pointer_cast<LmdbState>(state);

  CHECK_NOTNULL(lmdb_state.get());

  MDB_cursor*& cursor = lmdb_state->cursor_;

  MDB_val mdb_key;
  MDB_val mdb_val;
  CHECK_EQ(mdb_cursor_get(cursor, &mdb_key, &mdb_val, MDB_GET_CURRENT),
      MDB_SUCCESS);

  lmdb_state->kv_pair_ = make_pair(
    string(reinterpret_cast<char*>(mdb_key.mv_data), mdb_key.mv_size),
    string(reinterpret_cast<char*>(mdb_val.mv_data), mdb_val.mv_size));

  return lmdb_state->kv_pair_;
}

}  // namespace caffe
