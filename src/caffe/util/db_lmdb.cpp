/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef USE_LMDB
#include "caffe/util/db_lmdb.hpp"

#include <sys/stat.h>

#include <string>

namespace caffe { namespace db {

void LMDB::Open(const string& source, Mode mode) {
  MDB_CHECK(mdb_env_create(&mdb_env_));
  if (mode == NEW) {
    CHECK_EQ(mkdir(source.c_str(), 0744), 0) << "mkdir " << source << " failed";
  }
  int flags = 0;
  if (mode == READ) {
    flags = MDB_RDONLY | MDB_NOTLS;
#ifdef ALLOW_LMDB_NOLOCK
    flags |= MDB_NOLOCK;
#endif
  }
  int rc = mdb_env_open(mdb_env_, source.c_str(), flags, 0664);
  MDB_CHECK(rc);
  LOG(INFO) << "Opened lmdb " << source;
}

LMDBCursor* LMDB::NewCursor() {
  MDB_txn* mdb_txn;
  MDB_cursor* mdb_cursor;
  MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn));
  MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_));
  MDB_CHECK(mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor));
  return new LMDBCursor(mdb_txn, mdb_cursor);
}

LMDBTransaction* LMDB::NewTransaction() {
  return new LMDBTransaction(mdb_env_);
}

void LMDBTransaction::Put(const string& key, const string& value) {
  keys.push_back(key);
  values.push_back(value);
}

void LMDBTransaction::Commit() {
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;

  // Initialize MDB variables
  MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, 0, &mdb_txn));
  MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi));

  for (int i = 0; i < keys.size(); i++) {
    mdb_key.mv_size = keys[i].size();
    mdb_key.mv_data = const_cast<char*>(keys[i].data());
    mdb_data.mv_size = values[i].size();
    mdb_data.mv_data = const_cast<char*>(values[i].data());

    // Add data to the transaction
    int put_rc = mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0);
    if (put_rc == MDB_MAP_FULL) {
      // Out of memory - double the map size and retry
      mdb_txn_abort(mdb_txn);
      mdb_dbi_close(mdb_env_, mdb_dbi);
      DoubleMapSize();
      Commit();
      return;
    }
    // May have failed for some other reason
    MDB_CHECK(put_rc);
  }

  // Commit the transaction
  int commit_rc = mdb_txn_commit(mdb_txn);
  if (commit_rc == MDB_MAP_FULL) {
    // Out of memory - double the map size and retry
    mdb_dbi_close(mdb_env_, mdb_dbi);
    DoubleMapSize();
    Commit();
    return;
  }
  // May have failed for some other reason
  MDB_CHECK(commit_rc);

  // Cleanup after successful commit
  mdb_dbi_close(mdb_env_, mdb_dbi);
  keys.clear();
  values.clear();
}

void LMDBTransaction::DoubleMapSize() {
  struct MDB_envinfo current_info;
  MDB_CHECK(mdb_env_info(mdb_env_, &current_info));
  size_t new_size = current_info.me_mapsize * 2;
  DLOG(INFO) << "Doubling LMDB map size to " << (new_size>>20) << "MB ...";
  MDB_CHECK(mdb_env_set_mapsize(mdb_env_, new_size));
}

}  // namespace db
}  // namespace caffe
#endif  // USE_LMDB
