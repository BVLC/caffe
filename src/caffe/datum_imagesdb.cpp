#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <fstream>  // NOLINT(readability/streams)

#include "caffe/datum_imagesdb.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

void DatumImagesDB::Open() {
  if (*is_opened_ == false) {
    const string& source = param_.source();
    LOG(INFO) << "Opening imagesdb " << source;
    datum_database_.reset(new map<string, Datum>());
    switch (param_.mode()) {
    case DatumDBParameter_Mode_NEW: {
      file_.open(source.c_str(), ios::out | ios::trunc);
      CHECK(file_.good()) << "Could not create file: " << source;
      keys_.clear();
      batch_.clear();
      break;
    }
    case DatumDBParameter_Mode_WRITE: {
      file_.open(source.c_str(), ios::out | ios::app);
      CHECK(file_.good()) << "Could not open file: " << source;
      keys_.clear();
      batch_.clear();
      break;
    }
    case DatumDBParameter_Mode_READ: {
      string root_images = param_.root_images();
      bool cache_images = param_.cache_images();
      bool encode_images = param_.encode_images();
      file_.open(source.c_str(), ios::in);
      CHECK(file_.good()) << "File not found: " << source;
      keys_.clear();
      batch_.clear();
      std::string key;
      int label;
      while (file_ >> key >> label) {
        Datum datum;
        if (cache_images) {
          if (encode_images) {
            ReadFileToDatum(root_images + key, label, &datum);
          } else {
            ReadImageToDatum(root_images + key, label, &datum);
          }
        } else {
          datum.set_source(root_images + key);
          datum.set_label(label);
          datum.set_encoded(encode_images);
        }
        datum_database_->insert(make_pair(key, datum));
        keys_.push_back(key);
      }
      CHECK(!keys_.empty()) << "Error reading the file " << source;
      LOG(INFO) << "A total of " << datum_database_->size()
        << " images and " << keys_.size() << " keys.";
      file_.close();
      break;
    }
    default:
      LOG(FATAL) << "Unknown DB mode " << param_.mode();
    }
    *is_opened_ = true;
  }
}

void DatumImagesDB::Close() {
  if (*is_opened_ && is_opened_.unique()) {
    LOG(INFO) << "Closing imagesdb " << param_.source();
    batch_.clear();
    keys_.clear();
    datum_database_->clear();
    if (file_.is_open()) {
      file_.close();
    }
  }
}

bool DatumImagesDB::Get(const string& key, Datum* datum) {
  map<string, Datum>::iterator it = datum_database_->find(key);
  if (it != datum_database_->end()) {
    *datum = (*it).second;
    if (datum->data().size() == 0) {
      if (datum->encoded()) {
        ReadFileToDatum(datum->source(), datum->label(), datum);
      } else {
        ReadImageToDatum(datum->source(), datum->label(), datum);
      }
    }
    return (datum->data().size() > 0);
  } else {
    LOG(ERROR) << "key " << key << " not found";
    return false;
  }
}

void DatumImagesDB::Put(const string& key, const Datum& datum) {
  batch_.push_back(make_pair(key, datum));
}

void DatumImagesDB::Commit() {
  CHECK(param_.mode() != DatumDBParameter_Mode_READ);
  for (int i = 0; i < batch_.size(); ++i) {
    string key = batch_[i].first;
    Datum datum = batch_[i].second;
    keys_.push_back(key);
    (*datum_database_)[key] = datum;
    file_ << key << " " << datum.label() << std::endl;
  }
  batch_.clear();
}

DatumDBCursor* DatumImagesDB::NewCursor() {
  CHECK_EQ(param_.mode(), DatumDBParameter_Mode_READ)
    << "Only DatumDB in Mode_READ can create NewCursor";
  CHECK(*is_opened_);
  LOG(INFO) << "Creating NewCursor for " << param_.source();
  return new DatumImagesDBCursor(param_, datum_database_, keys_);
}

// DatumImagesDBCursor
bool DatumImagesDBCursor::Valid() {
  return (!keys_.empty() && read_it_ != keys_.end());
}

void DatumImagesDBCursor::SeekToFirst() {
  CHECK(!keys_.empty());
  if (param_.shuffle_images()) {
    ShuffleKeys();
  }
  read_it_ = keys_.begin();
}

void DatumImagesDBCursor::Next() {
  CHECK(Valid());
  ++read_it_;
  if (read_it_ == keys_.end()) {
    if (param_.loop()) {
      SeekToFirst();
    } else {
      LOG(ERROR) << "Reached the end and not looping.";
    }
  }
}

string DatumImagesDBCursor::key() {
  CHECK(Valid());
  return *read_it_;
}

Datum DatumImagesDBCursor::value() {
  CHECK(Valid());
  string key = *read_it_;
  map<string, Datum>::iterator it = datum_database_->find(key);
  Datum datum = (*it).second;
  if (datum.data().size() == 0) {
    if (datum.encoded()) {
      ReadFileToDatum(datum.source(), datum.label(), &datum);
    } else {
      ReadImageToDatum(datum.source(), datum.label(), &datum);
    }
  }
  return datum;
}

void DatumImagesDBCursor::ShuffleKeys() {
  // randomly shuffle the keys
  LOG(INFO) << "Shuffling the keys";
  shuffle(keys_.begin(), keys_.end());
}

REGISTER_DATUMDB_CLASS("imagesdb", DatumImagesDB);
}  // namespace caffe
