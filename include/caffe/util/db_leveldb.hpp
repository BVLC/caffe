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

#ifdef USE_LEVELDB
#ifndef CAFFE_UTIL_DB_LEVELDB_HPP
#define CAFFE_UTIL_DB_LEVELDB_HPP

#include <string>
#include <utility>

#include "leveldb/db.h"
#include "leveldb/write_batch.h"

#include "caffe/util/db.hpp"

namespace caffe { namespace db {

class LevelDBCursor : public Cursor {
 public:
  explicit LevelDBCursor(leveldb::Iterator* iter)
    : iter_(iter) { SeekToFirst(); }
  ~LevelDBCursor() { delete iter_; }
  virtual void SeekToFirst() { iter_->SeekToFirst(); }
  virtual void Next() { iter_->Next(); }
  virtual string key() { return iter_->key().ToString(); }
  virtual string value() { return iter_->value().ToString(); }
  virtual std::pair<void*, size_t> valuePointer() {
    CHECK(false) << "Function valuePointer not implemented in LevelDBCursor";
    return std::pair<void*, size_t>{};
  }
  virtual bool valid() { return iter_->Valid(); }

 private:
  leveldb::Iterator* iter_;
};

class LevelDBTransaction : public Transaction {
 public:
  explicit LevelDBTransaction(leveldb::DB* db) : db_(db) { CHECK_NOTNULL(db_); }
  virtual void Put(const string& key, const string& value) {
    batch_.Put(key, value);
  }
  virtual void Commit() {
    leveldb::Status status = db_->Write(leveldb::WriteOptions(), &batch_);
    CHECK(status.ok()) << "Failed to write batch to leveldb "
                       << std::endl << status.ToString();
  }

 private:
  leveldb::DB* db_;
  leveldb::WriteBatch batch_;

  DISABLE_COPY_AND_ASSIGN(LevelDBTransaction);
};

class LevelDB : public DB {
 public:
  LevelDB() : db_(NULL) { }
  virtual ~LevelDB() { Close(); }
  virtual void Open(const string& source, Mode mode);
  virtual void Close() {
    if (db_ != NULL) {
      delete db_;
      db_ = NULL;
    }
  }
  virtual LevelDBCursor* NewCursor() {
    return new LevelDBCursor(db_->NewIterator(leveldb::ReadOptions()));
  }
  virtual LevelDBTransaction* NewTransaction() {
    return new LevelDBTransaction(db_);
  }

 private:
  leveldb::DB* db_;
};


}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_LEVELDB_HPP
#endif  // USE_LEVELDB
