#ifndef CAFFE_DATUMDB_HPP
#define CAFFE_DATUMDB_HPP

#include <string>

#include "leveldb/db.h"
#include "lmdb.h"

#include "caffe/common.hpp"
#include "caffe/datum_DB_factory.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

class DatumDBCursor {
 public:
  explicit DatumDBCursor(const DatumDBParameter& param)
    : param_(param) {}
  ~DatumDBCursor() {
    LOG(INFO) << "Closing DatumDBCursor on " << param_.source();
  }

  virtual bool Valid() = 0;
  virtual void SeekToFirst() = 0;
  virtual void Next() = 0;
  virtual string key() = 0;
  virtual Datum value() = 0;

 protected:
  DatumDBParameter param_;
  DISABLE_COPY_AND_ASSIGN(DatumDBCursor);
};

class DatumDB {
 public:
  explicit DatumDB(const DatumDBParameter& param)
    : param_(param),
      is_opened_(new bool(false)) {}
  virtual ~DatumDB() {
    if (is_opened_.unique()) {
      DatumDBRegistry::RemoveSource(param_.source());
    }
  }

  virtual bool Get(const string& key, Datum* value) = 0;
  virtual void Put(const string& key, const Datum& value) = 0;
  virtual void Commit() = 0;
  virtual DatumDBCursor* NewCursor() = 0;

 protected:
  virtual void Open() = 0;
  virtual void Close() = 0;

  DatumDBParameter param_;
  shared_ptr<bool> is_opened_;
  DISABLE_COPY_AND_ASSIGN(DatumDB);
};

class DatumLevelDB : public DatumDB {
 public:
  explicit DatumLevelDB(const DatumDBParameter& param)
    : DatumDB(param) { Open(); }
  virtual ~DatumLevelDB() { Close(); }

  virtual bool Get(const string& key, Datum* datum);
  virtual void Put(const string& key, const Datum& datum);
  virtual void Commit();
  virtual DatumDBCursor* NewCursor();

 protected:
  virtual void Open();
  virtual void Close();

  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::WriteBatch> batch_;
};

class DatumLevelDBCursor : public DatumDBCursor {
 public:
  explicit DatumLevelDBCursor(const DatumDBParameter& param,
                              leveldb::Iterator* iter)
    : DatumDBCursor(param),
    iter_(iter) { CHECK_NOTNULL(iter_); SeekToFirst(); }
  ~DatumLevelDBCursor() {
    LOG(INFO) << "Closing DatumLevelDBCursor";
  }
  virtual bool Valid();
  virtual void SeekToFirst();
  virtual void Next();
  virtual string key();
  virtual Datum value();

 protected:
  leveldb::Iterator* iter_;
};

class DatumLMDB : public DatumDB {
 public:
  explicit DatumLMDB(const DatumDBParameter& param)
    : DatumDB(param) { Open(); }
  virtual ~DatumLMDB() { Close(); }

  virtual bool Get(const string& key, Datum* datum);
  virtual void Put(const string& key, const Datum& datum);
  virtual void Commit();
  virtual DatumDBCursor* NewCursor();

 protected:
  virtual void Open();
  virtual void Close();

  MDB_env* mdb_env_;
  MDB_txn* mdb_txn_;
  MDB_dbi mdb_dbi_;
};

class DatumLMDBCursor : public DatumDBCursor {
 public:
  explicit DatumLMDBCursor(const DatumDBParameter& param,
                           MDB_cursor* mdb_cursor)
    : DatumDBCursor(param),
    mdb_cursor_(mdb_cursor) { CHECK_NOTNULL(mdb_cursor_); SeekToFirst(); }
  ~DatumLMDBCursor() {
    LOG(INFO) << "Closing DatumLMDBCursor";
    mdb_cursor_close(mdb_cursor_);
  }
  virtual bool Valid();
  virtual void SeekToFirst();
  virtual void Next();
  virtual string key();
  virtual Datum value();

 protected:
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
  int mdb_status_;
};


}  // namespace caffe

#endif  // CAFFE_DATUMDB_HPP
