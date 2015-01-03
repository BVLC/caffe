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

  virtual shared_ptr<Generator> NewGenerator();
  virtual bool Get(const string& key, Datum* datum);
  virtual void Put(const string& key, const Datum& datum);
  virtual void Commit();

 protected:
  explicit DatumImagesDB(const DatumImagesDB& other)
    : DatumDB(other),
    cache_images_(other.cache_images_),
    encode_images_(other.encode_images_),
    keys_(other.keys_),
    datum_database_(other.datum_database_) {}
  virtual void Open();
  virtual void Close();
  virtual bool Valid();
  virtual bool Next();
  virtual bool Reset();
  virtual bool Current(string* key, Datum* datum);
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

#endif  // CAFFE_IMAGESDB_HPP
