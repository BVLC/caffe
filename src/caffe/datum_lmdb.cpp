#include <sys/stat.h>

#include <string>

#include "caffe/datum_DB.hpp"

namespace caffe {

void DatumLMDB::Open() {
  LOG(INFO) << "Opening lmdb " << param_.source();
  mdb_status_ = mdb_env_create(&mdb_env_);
  CHECK_EQ(mdb_status_, MDB_SUCCESS) << mdb_strerror(mdb_status_);
  switch (param_.mode()) {
  case DatumDBParameter_Mode_NEW:
    CHECK_EQ(mkdir(param_.source().c_str(), 0744), 0)
      << "mkdir " << param_.source() << "failed";
  case DatumDBParameter_Mode_WRITE:
    mdb_status_ = mdb_env_set_mapsize(mdb_env_, param_.mdb_env_mapsize());
    CHECK_EQ(mdb_status_, MDB_SUCCESS) << mdb_strerror(mdb_status_);
    mdb_status_ = mdb_env_open(mdb_env_, param_.source().c_str(), 0, 0664);
    CHECK_EQ(mdb_status_, MDB_SUCCESS) << mdb_strerror(mdb_status_);
    mdb_status_ = mdb_txn_begin(mdb_env_, NULL, 0, &mdb_txn_);
    CHECK_EQ(mdb_status_, MDB_SUCCESS) << mdb_strerror(mdb_status_);
    mdb_status_ = mdb_dbi_open(mdb_txn_, NULL, MDB_CREATE, &mdb_dbi_);
    CHECK_EQ(mdb_status_, MDB_SUCCESS) << mdb_strerror(mdb_status_);
    break;
  case DatumDBParameter_Mode_READ:
    mdb_status_ = mdb_env_set_mapsize(mdb_env_, 1);
    CHECK_EQ(mdb_status_, MDB_SUCCESS) << mdb_strerror(mdb_status_);
    mdb_status_ = mdb_env_set_maxreaders(mdb_env_, param_.mdb_env_maxreaders());
    CHECK_EQ(mdb_status_, MDB_SUCCESS) << mdb_strerror(mdb_status_);
    mdb_status_ = mdb_env_open(mdb_env_, param_.source().c_str(),
                  MDB_RDONLY|MDB_NOTLS, 0664);
    CHECK_EQ(mdb_status_, MDB_SUCCESS) << mdb_strerror(mdb_status_);
    mdb_status_ = mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_);
    CHECK_EQ(mdb_status_, MDB_SUCCESS) << mdb_strerror(mdb_status_);
    mdb_status_ = mdb_dbi_open(mdb_txn_, NULL, 0, &mdb_dbi_);
    CHECK_EQ(mdb_status_, MDB_SUCCESS) << mdb_strerror(mdb_status_);
    mdb_status_ = mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_);
    CHECK_EQ(mdb_status_, MDB_SUCCESS) << mdb_strerror(mdb_status_);
    CHECK(SeekToFirst()) << "Failed SeekToFirst";
    break;
  default:
    LOG(FATAL) << "Unknown DB Mode " << param_.mode();
  }
}

void DatumLMDB::Close() {
  LOG(INFO) << "Closing lmdb " << param_.source();
  if (param_.mode() == DatumDBParameter_Mode_READ) {
    mdb_cursor_close(mdb_cursor_);
  }
  mdb_close(mdb_env_, mdb_dbi_);
  mdb_txn_abort(mdb_txn_);
  mdb_env_close(mdb_env_);
}

bool DatumLMDB::Next() {
  CHECK(mdb_cursor_);
  mdb_status_ = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT);
  if (mdb_status_ != MDB_SUCCESS) {
    if (param_.loop()) {
      SeekToFirst();
      DLOG(INFO) << "Reached the end and looping.";
    } else {
      LOG(ERROR) << "Reached the end and not looping.";
    }
  }
  return Valid();
}

bool DatumLMDB::Prev() {
  CHECK(mdb_cursor_);
  mdb_status_ = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_PREV);
  if (mdb_status_ != MDB_SUCCESS) {
    if (param_.loop()) {
      SeekToLast();
      DLOG(INFO) << "Reached the beginning and looping.";
    } else {
      LOG(ERROR) << "Reached the beginning and not looping.";
    }
  }
  return Valid();
}

bool DatumLMDB::SeekToFirst() {
  CHECK(mdb_cursor_);
  mdb_status_ = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST);
  return Valid();
}

bool DatumLMDB::SeekToLast() {
  CHECK(mdb_cursor_);
  mdb_status_ = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_LAST);
  return Valid();
}

bool DatumLMDB::Valid() {
  if (mdb_status_ == MDB_SUCCESS) {
    return (mdb_value_.mv_data && mdb_value_.mv_size > 0
            && mdb_key_.mv_data && mdb_key_.mv_size > 0);
  } else {
    LOG(ERROR) << mdb_strerror(mdb_status_);
    return false;
  }
}

bool DatumLMDB::Current(string* key, Datum* datum) {
  if (Valid()) {
    key->assign(string(reinterpret_cast<char*>(mdb_key_.mv_data),
                      mdb_key_.mv_size));
    datum->ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
  }
  return Valid();
}

bool DatumLMDB::Get(const string& key, Datum* datum) {
  CHECK(mdb_cursor_);
  string aux_key(key);
  mdb_key_.mv_size = aux_key.size();
  mdb_key_.mv_data = reinterpret_cast<void*>(&aux_key[0]);
  mdb_status_ = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_SET);
  if (mdb_status_ == MDB_SUCCESS) {
    datum->ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    return true;
  } else {
    LOG(ERROR) << "key " << key << " not found";
    return false;
  }
}

void DatumLMDB::Put(const string& key, const Datum& datum) {
  CHECK(param_.mode() != DatumDBParameter_Mode_READ);
  string aux_key(key);
  mdb_key_.mv_size = aux_key.size();
  mdb_key_.mv_data = reinterpret_cast<void*>(&aux_key[0]);
  string value;
  datum.SerializeToString(&value);
  mdb_value_.mv_size = value.size();
  mdb_value_.mv_data = reinterpret_cast<void*>(&value[0]);
  mdb_status_ = mdb_put(mdb_txn_, mdb_dbi_, &mdb_key_, &mdb_value_, 0);
  CHECK_EQ(mdb_status_, MDB_SUCCESS) << mdb_strerror(mdb_status_);
}

void DatumLMDB::Commit() {
  CHECK(param_.mode() != DatumDBParameter_Mode_READ);
  mdb_status_ = mdb_txn_commit(mdb_txn_);
  CHECK_EQ(mdb_status_, MDB_SUCCESS) << mdb_strerror(mdb_status_);
  mdb_status_ = mdb_txn_begin(mdb_env_, NULL, 0, &mdb_txn_);
  CHECK_EQ(mdb_status_, MDB_SUCCESS) << mdb_strerror(mdb_status_);
}

}  // namespace caffe
