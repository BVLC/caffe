#ifndef CAFFE_IMAGESDB_HPP
#define CAFFE_IMAGESDB_HPP

#include <fstream>  // NOLINT(readability/streams)
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "caffe/datum_DB.hpp"

namespace caffe {

class DatumImagesDB : public DatumDB {
 public:
  explicit DatumImagesDB(const DatumDBParameter& param)
    : DatumDB(param) { Open(); }
  virtual ~DatumImagesDB() { Close(); }

  virtual bool Get(const string& key, Datum* datum);
  virtual void Put(const string& key, const Datum& datum);
  virtual void Commit();
  virtual DatumDBCursor* NewCursor();

 protected:
  virtual void Open();
  virtual void Close();

  std::fstream file_;
  shared_ptr<map<string, Datum> > datum_database_;
  vector<string> keys_;
  vector<pair<string, Datum> > batch_;
};

class DatumImagesDBCursor : public DatumDBCursor {
 public:
  explicit DatumImagesDBCursor(const DatumDBParameter& param,
    shared_ptr<map<string, Datum> > datum_database, vector<string> keys)
    : DatumDBCursor(param),
    datum_database_(datum_database),
    keys_(keys) { SeekToFirst(); }
  ~DatumImagesDBCursor() {
    LOG(INFO) << "Closing DatumImagesDBCursor";
  }

  virtual bool Valid();
  virtual void SeekToFirst();
  virtual void Next();
  virtual string key();
  virtual Datum value();

 protected:
  virtual void ShuffleKeys();

  shared_ptr<map<string, Datum> > datum_database_;
  vector<string> keys_;
  vector<string>::iterator read_it_;
};

}  // namespace caffe

#endif  // CAFFE_IMAGESDB_HPP
