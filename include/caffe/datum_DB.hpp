#ifndef CAFFE_DATUMDB_HPP
#define CAFFE_DATUMDB_HPP

#include <algorithm>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "leveldb/db.h"
#include "lmdb.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

class DatumDB {
 public:
  explicit DatumDB(const DatumDBParameter& param)
    : param_(param),
      is_opened_(false) {}
  virtual ~DatumDB() {}

  class Generator {
   public:
    Generator(shared_ptr<DatumDB> datumdb)
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

  static shared_ptr<DatumDB> GetDatumDB(const DatumDBParameter& param);
  static shared_ptr<Generator> GetGenerator(const DatumDBParameter& param);

  virtual shared_ptr<Generator> GetGenerator() = 0;

  virtual bool Get(const string& key, Datum* datum) = 0;
  virtual void Put(const string& key, const Datum& datum) = 0;
  virtual void Commit() = 0;

 protected:
  virtual bool Valid() = 0;
  virtual bool Reset() = 0;
  virtual bool Next() = 0;
  virtual bool Current(string* key, Datum* datum) = 0;
  virtual void Open() = 0;
  virtual void Close() = 0;

  DatumDBParameter param_;
  bool is_opened_;

  DISABLE_COPY_AND_ASSIGN(DatumDB);
};

class DatumLevelDB : public DatumDB {
 public:
  explicit DatumLevelDB(const DatumDBParameter& param)
    : DatumDB(param) {}
  virtual ~DatumLevelDB() { Close(); }

  virtual shared_ptr<Generator> GetGenerator();
  virtual bool Get(const string& key, Datum* datum);
  virtual void Put(const string& key, const Datum& datum);
  virtual void Commit();

 protected:
  virtual bool Valid();
  virtual bool Next();
  virtual bool Reset();
  virtual bool Current(string* key, Datum* datum);
  virtual void Open();
  virtual void Close();

  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  shared_ptr<leveldb::WriteBatch> batch_;
};


class DatumLMDB : public DatumDB {
 public:
  explicit DatumLMDB(const DatumDBParameter& param)
    : DatumDB(param) {}
  virtual ~DatumLMDB() { Close(); }

  virtual shared_ptr<Generator> GetGenerator();
  virtual bool Get(const string& key, Datum* datum);
  virtual void Put(const string& key, const Datum& datum);
  virtual void Commit();

 protected:
  virtual bool Valid();
  virtual bool Next();
  virtual bool Reset();
  virtual bool Current(string* key, Datum* datum);
  virtual void Open();
  virtual void Close();

  MDB_env* mdb_env_;
  MDB_txn* mdb_txn_;
  MDB_dbi mdb_dbi_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
  int mdb_status_;
};

class DatumImagesDB : public DatumDB {
 public:
  explicit DatumImagesDB(const DatumDBParameter& param)
    : DatumDB(param) {}
  virtual ~DatumImagesDB() { Close(); }

  virtual shared_ptr<Generator> GetGenerator();
  virtual bool Get(const string& key, Datum* datum);
  virtual void Put(const string& key, const Datum& datum);
  virtual void Commit();

 protected:
  virtual bool Valid();
  virtual bool Next();
  virtual bool Reset();
  virtual bool Current(string* key, Datum* datum);
  virtual void Open();
  virtual void Close();
  virtual void ShuffleKeys();

  std::fstream file_;
  string root_images_;
  bool cache_images_;
  bool encode_images_;

  vector<string> keys_;
  vector<string>::iterator read_it_;
  shared_ptr<map<string, Datum> > datum_database_;
  vector<pair<string, Datum> > batch_;
};

}  // namespace caffe

#endif  // CAFFE_DATUMDB_HPP
