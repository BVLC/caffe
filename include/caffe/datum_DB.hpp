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
    : param_(param) {}
  virtual ~DatumDB() {}
  static shared_ptr<DatumDB> GetDatumDB(const DatumDBParameter& param);
  static DatumDBParameter_Backend GetBackend(const string& backend);

  virtual bool Next() = 0;
  virtual bool Prev() = 0;
  virtual bool SeekToFirst() = 0;
  virtual bool SeekToLast() = 0;
  virtual bool Valid() = 0;
  virtual bool Current(string* key, Datum* datum) = 0;
  virtual bool Get(const string& key, Datum* datum) = 0;
  virtual void Put(const string& key, const Datum& datum) = 0;
  virtual void Commit() = 0;

  virtual Datum CurrentDatum() {
    string key;
    Datum datum;
    CHECK(Current(&key, &datum)) << "Current not valid";
    return datum;
  }
  virtual string CurrentKey() {
    string key;
    Datum datum;
    CHECK(Current(&key, &datum)) << "Current not valid";
    return key;
  }

 protected:
  virtual void Open() = 0;
  virtual void Close() = 0;

  DatumDBParameter param_;
};


class DatumLevelDB : public DatumDB {
 public:
  explicit DatumLevelDB(const DatumDBParameter& param)
    : DatumDB(param) { Open(); }
  virtual ~DatumLevelDB() { Close(); }

  virtual bool Next();
  virtual bool Prev();
  virtual bool SeekToFirst();
  virtual bool SeekToLast();
  virtual bool Valid();
  virtual bool Current(string* key, Datum* datum);
  virtual bool Get(const string& key, Datum* datum);
  virtual void Put(const string& key, const Datum& datum);
  virtual void Commit();

 protected:
  virtual void Open();
  virtual void Close();

  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  leveldb::WriteBatch* batch_;
};


class DatumLMDB : public DatumDB {
 public:
  explicit DatumLMDB(const DatumDBParameter& param)
    : DatumDB(param) { Open(); }
  virtual ~DatumLMDB() { Close(); }

  virtual bool Next();
  virtual bool Prev();
  virtual bool SeekToFirst();
  virtual bool SeekToLast();
  virtual bool Valid();
  virtual bool Current(string* key, Datum* datum);
  virtual bool Get(const string& key, Datum* datum);
  virtual void Put(const string& key, const Datum& datum);
  virtual void Commit();

 protected:
  virtual void Open();
  virtual void Close();

  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
  int mdb_status_;
};

class DatumImagesDB : public DatumDB {
 public:
  explicit DatumImagesDB(const DatumDBParameter& param)
    : DatumDB(param) { Open(); }
  virtual ~DatumImagesDB() { Close(); }

  virtual bool Next();
  virtual bool Prev();
  virtual bool SeekToFirst();
  virtual bool SeekToLast();
  virtual bool Valid();
  virtual bool Current(string* key, Datum* datum);
  virtual bool Get(const string& key, Datum* datum);
  virtual void Put(const string& key, const Datum& datum);
  virtual void Commit();

 protected:
  virtual void Open();
  virtual void Close();

  std::fstream file_;
  string root_images_;
  bool cache_images_;
  bool encode_images_;

  vector<string> keys_;
  vector<string>::iterator read_it_;
  map<string, Datum> datum_database_;
  vector<pair<string, Datum> > batch_;
};

}  // namespace caffe

#endif  // CAFFE_DATUMDB_HPP
