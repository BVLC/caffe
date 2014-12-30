#include <map>
#include <string>
#include <vector>

#include <fstream>  // NOLINT(readability/streams)

#include "caffe/datum_DB.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

void DatumImagesDB::Open() {
  CHECK(!is_opened_) << "Already opened";
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
    root_images_ = param_.root_images();
    cache_images_ = param_.cache_images();
    encode_images_ = param_.encode_images();
    file_.open(source.c_str(), ios::in);
    CHECK(file_.good()) << "File not found: " << source;
    keys_.clear();
    batch_.clear();
    std::string key;
    int label;
    while (file_ >> key >> label) {
      Datum datum;
      if (cache_images_) {
        if (encode_images_) {
          ReadFileToDatum(root_images_ + key, label, &datum);
        } else {
          ReadImageToDatum(root_images_ + key, label, &datum);
        }
      } else {
        datum.set_source(root_images_ + key);
        datum.set_label(label);
      }
      datum_database_->insert(make_pair(key, datum));
      keys_.push_back(key);
    }
    CHECK(!keys_.empty()) << "Error reading the file " << source;
    LOG(INFO) << "A total of " << datum_database_->size()
      << " images and " << keys_.size() << " keys.";
    break;
  }
  default:
    LOG(FATAL) << "Unknown DB mode " << param_.mode();
  }
  is_opened_ = true;
}

void DatumImagesDB::Close() {
  batch_.clear();
  keys_.clear();
  if (is_opened_) {
    LOG(INFO) << "Closing imagesdb " << param_.source();
    datum_database_->clear();
    datum_database_.reset();
    file_.close();
  } else {
    LOG(INFO) << "Closing Generator on " << param_.source();
  }
}

shared_ptr<DatumDB::Generator> DatumImagesDB::GetGenerator() {
  CHECK_EQ(param_.mode(),DatumDBParameter_Mode_READ)
    << "Only DatumDB in Mode_READ can use GetGenerator";
  LOG(INFO) << "Creating Generator for " << param_.source();
  DatumImagesDB* datumdb = new DatumImagesDB(param_);
  datumdb->datum_database_ = datum_database_;
  datumdb->keys_ = keys_;
  shared_ptr<DatumDB::Generator> generator;
  generator.reset(new DatumDB::Generator(shared_ptr<DatumDB>(datumdb)));
  return generator;
}

bool DatumImagesDB::Valid() {
  return (!keys_.empty() && read_it_ != keys_.end());
}

bool DatumImagesDB::Next() {
  if (Valid()) {
    ++read_it_;
    if (read_it_ == keys_.end()) {
      if (param_.loop()) {
        return Reset();
      } else {
        LOG(ERROR) << "Reached the end and not looping.";
      }
    }
  }
  return Valid();
}

bool DatumImagesDB::Reset() {
  if (!keys_.empty()) {
    if (param_.shuffle_images()) {
      ShuffleKeys();
    }
    read_it_ = keys_.begin();
  }
  return Valid();
}

bool DatumImagesDB::Current(string* key, Datum* datum) {
  if (Valid()) {
    *key = *read_it_;
    return Get(*key, datum);
  } else {
    return false;
  }
}

void DatumImagesDB::ShuffleKeys() {
  // randomly shuffle the keys
  LOG(INFO) << "Shuffling the keys";
  shuffle(keys_.begin(), keys_.end());
}

bool DatumImagesDB::Get(const string& key, Datum* datum) {
  map<string, Datum>::iterator it = datum_database_->find(key);
  if (it != datum_database_->end()) {
    *datum = (*it).second;
    if (!cache_images_) {
      if (encode_images_) {
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

}  // namespace caffe
