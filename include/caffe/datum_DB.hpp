#ifndef CAFFE_DATUMDB_HPP
#define CAFFE_DATUMDB_HPP

#include <string>

#include "leveldb/db.h"
#include "lmdb.h"

#include "caffe/common.hpp"
#include "caffe/datum_DB_factory.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

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

  class Generator {
   public:
    explicit Generator(shared_ptr<DatumDB> datumdb)
      : datumdb_(datumdb) { CHECK_NOTNULL(datumdb.get());
          CHECK(Reset()); }
    bool Valid() { return datumdb_->Valid(); }
    bool Reset() { return datumdb_->Reset(); }
    bool Next() { return datumdb_->Next(); }
    bool Current(string* key, Datum* datum) {
      return datumdb_->Current(key, datum);
    }
    bool Current(Datum* datum) {
      string key;
      return datumdb_->Current(&key, datum);
    }
   private:
    shared_ptr<DatumDB> datumdb_;
    DISABLE_COPY_AND_ASSIGN(Generator);
  };

  virtual shared_ptr<Generator> NewGenerator() = 0;
  virtual bool Get(const string& key, Datum* datum) = 0;
  virtual void Put(const string& key, const Datum& datum) = 0;
  virtual void Commit() = 0;

 protected:
  explicit DatumDB(const DatumDB& other)
    : param_(other.param_),
      is_opened_(other.is_opened_) {}
  virtual void Open() = 0;
  virtual void Close() = 0;
  virtual bool Valid() = 0;
  virtual bool Reset() = 0;
  virtual bool Next() = 0;
  virtual bool Current(string* key, Datum* datum) = 0;

  DatumDBParameter param_;
  shared_ptr<bool> is_opened_;
};

class DatumLevelDB : public DatumDB {
 public:
  explicit DatumLevelDB(const DatumDBParameter& param)
    : DatumDB(param) { Open(); }
  virtual ~DatumLevelDB() { Close(); }

  virtual shared_ptr<Generator> NewGenerator();
  virtual bool Get(const string& key, Datum* datum);
  virtual void Put(const string& key, const Datum& datum);
  virtual void Commit();

 protected:
  explicit DatumLevelDB(const DatumLevelDB& other)
    : DatumDB(other),
    db_(other.db_) {}
  virtual void Open();
  virtual void Close();
  virtual bool Valid();
  virtual bool Next();
  virtual bool Reset();
  virtual bool Current(string* key, Datum* datum);

  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  shared_ptr<leveldb::WriteBatch> batch_;
};


class DatumLMDB : public DatumDB {
 public:
  explicit DatumLMDB(const DatumDBParameter& param)
    : DatumDB(param) { Open(); }
  virtual ~DatumLMDB() { Close(); }

  virtual shared_ptr<Generator> NewGenerator();
  virtual bool Get(const string& key, Datum* datum);
  virtual void Put(const string& key, const Datum& datum);
  virtual void Commit();

 protected:
  explicit DatumLMDB(const DatumLMDB& other)
    : DatumDB(other),
    mdb_env_(other.mdb_env_),
    mdb_txn_(other.mdb_txn_),
    mdb_dbi_(other.mdb_dbi_) {}
  virtual void Open();
  virtual void Close();
  virtual bool Valid();
  virtual bool Next();
  virtual bool Reset();
  virtual bool Current(string* key, Datum* datum);

  MDB_env* mdb_env_;
  MDB_txn* mdb_txn_;
  MDB_dbi mdb_dbi_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
  int mdb_status_;
};

}  // namespace caffe

#endif  // CAFFE_DATUMDB_HPP
