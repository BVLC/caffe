#include <string>
#include <vector>

#include <fstream>  // NOLINT(readability/streams)

#include "caffe/datum_DB.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

void DatumImagesDB::Open() {
  const string& source = param_.source();
  LOG(INFO) << "Opening imagesdb " << source;
  switch (param_.mode()) {
  case DatumDBParameter_Mode_NEW: {
    file_.open(source.c_str(), ios::out | ios::trunc);
    CHECK(file_.good()) << "Could not create file: " << source;
    datum_database_.clear();
    keys_.clear();
    batch_.clear();
    break;
  }
  case DatumDBParameter_Mode_WRITE: {
    file_.open(source.c_str(), ios::out | ios::app);
    CHECK(file_.good()) << "Could not open file: " << source;
    datum_database_.clear();
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
    datum_database_.clear();
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
      datum_database_[key] = datum;
      keys_.push_back(key);
    }
    CHECK(!keys_.empty()) << "Error reading the file " << source;
    if (param_.shuffle_images()) {
      // randomly shuffle data
      LOG(INFO) << "Shuffling the keys";
      shuffle(keys_.begin(), keys_.end());
    }
    CHECK(SeekToFirst()) << "Failed SeekToFirst";
    LOG(INFO) << "A total of " << keys_.size() << " elements.";
    break;
  }
  default:
    LOG(FATAL) << "Unknown DB mode " << param_.mode();
  }
}

void DatumImagesDB::Close() {
  LOG(INFO) << "Closing imagesdb " << param_.source();
  batch_.clear();
  keys_.clear();
  datum_database_.clear();
  file_.close();
}

bool DatumImagesDB::Next() {
  if (Valid()) {
    ++read_it_;
    if (read_it_ == keys_.end()) {
      if (param_.loop()) {
        if (param_.shuffle_images()) {
          // randomly shuffle data
          LOG(INFO) << "Shuffling the keys";
          shuffle(keys_.begin(), keys_.end());
        }
        LOG(INFO) << "Reached the end and looping.";
        SeekToFirst();
      } else {
        LOG(ERROR) << "Reached the end and not looping.";
      }
    }
  }
  return Valid();
}

bool DatumImagesDB::Prev() {
  if (Valid()) {
    if (read_it_ == keys_.begin()) {
      if (param_.loop()) {
        if (param_.shuffle_images()) {
          // randomly shuffle data
          LOG(INFO) << "Shuffling the keys";
          shuffle(keys_.begin(), keys_.end());
        }
        LOG(INFO) << "Reached the beginning and looping.";
        SeekToLast();
      } else {
        LOG(ERROR) << "Reached the beginning and not looping.";
        read_it_ = keys_.end();
      }
    } else {
      --read_it_;
    }
  }
  return Valid();
}

bool DatumImagesDB::SeekToFirst() {
  if (!keys_.empty()) {
    read_it_ = keys_.begin();  
  }
  return Valid();
}

bool DatumImagesDB::SeekToLast() {
  read_it_ = keys_.end();
  if (!keys_.empty()) {
    --read_it_;
  }
  return Valid();
}

bool DatumImagesDB::Valid() {
  return (!keys_.empty() && read_it_ != keys_.end());
}

bool DatumImagesDB::Current(string* key, Datum* datum) {
  if (Valid()) {
    *key = *read_it_;
    return Get(*key, datum);
  } else {
    return false;
  }
}

bool DatumImagesDB::Get(const string& key, Datum* datum) {
  map<string, Datum>::iterator it = datum_database_.find(key);
  if (it != datum_database_.end()) {
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
    datum_database_[key] = datum;
    file_ << key << " " << datum.label() << std::endl;
  }
  batch_.clear();
}

}  // namespace caffe
