#include <string>

#include "leveldb/write_batch.h"

#include "caffe/datum_DB.hpp"

namespace caffe {

void DatumLevelDB::Open() {
  LOG(INFO) << "Opening leveldb " << param_.source();
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.block_size = 8192;
  options.write_buffer_size = 268435456;
  options.max_open_files = 100;
  switch (param_.mode()) {
  case DatumDBParameter_Mode_NEW:
    options.error_if_exists = true;
    options.create_if_missing = true;
    batch_ = new leveldb::WriteBatch();
    break;
  case DatumDBParameter_Mode_WRITE:
    options.error_if_exists = false;
    options.create_if_missing = true;
    batch_ = new leveldb::WriteBatch();
    break;
  case DatumDBParameter_Mode_READ:
    options.error_if_exists = false;
    options.create_if_missing = false;
    batch_ = NULL;
    break;
  default:
    LOG(FATAL) << "Unknown DB Mode " << param_.mode();
  }
  leveldb::Status status = leveldb::DB::Open(options,
                            param_.source(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb " << param_.source()
                     << std::endl << status.ToString();
  db_.reset(db_temp);
  if (param_.mode() == DatumDBParameter_Mode_READ) {
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    CHECK(SeekToFirst()) << "Failed SeekToFirst";
  }
}

void DatumLevelDB::Close() {
  LOG(INFO) << "Closing leveldb " << param_.source();
  if (batch_ != NULL) {
    delete batch_;
  }
  iter_.reset();
  db_.reset();
}

bool DatumLevelDB::Next() {
  if (Valid()) {
    iter_->Next();
    if (!iter_->Valid()) {
      if (param_.loop()) {
        // We have reached the end. Restart from the first.
        LOG(INFO) << "Reached the end and looping.";
        iter_->SeekToFirst();
      } else {
        LOG(ERROR) << "Reached the end and not looping.";
      }
    }
  }
  return Valid();
}

bool DatumLevelDB::Prev() {
  if (Valid()) {
    iter_->Prev();
    if (!iter_->Valid()) {
      if (param_.loop()) {
        // We have reached the end. Restart from the first.
        LOG(INFO) << "Reached the beginning and looping.";
        iter_->SeekToLast();
      } else {
        LOG(ERROR) << "Reached the beginning and not looping.";
      }
    }
  }   
  return Valid();
}

bool DatumLevelDB::SeekToFirst() {
  CHECK(iter_);
  iter_->SeekToFirst();
  return Valid();
}

bool DatumLevelDB::SeekToLast() {
  CHECK(iter_);
  iter_->SeekToLast();
  return Valid();
}

bool DatumLevelDB::Valid() {
  CHECK(iter_);
  return (iter_->Valid());
}

bool DatumLevelDB::Current(string* key, Datum* datum) {
  if (Valid()) {
    key->assign(iter_->key().ToString());
    datum->ParseFromString(iter_->value().ToString());
    return true;
  } else {
    return false;
  }
}

bool DatumLevelDB::Get(const string& key, Datum* datum) {
  CHECK(iter_);
  leveldb::Slice slice_key = key;
  iter_->Seek(slice_key);
  if (iter_->Valid() && iter_->key() == slice_key) {
    datum->ParseFromString(iter_->value().ToString());
    return true;
  } else {
    LOG(ERROR) << "key " << key << " not found";
    return false;
  }
}

void DatumLevelDB::Put(const string& key, const Datum& datum) {
  CHECK(param_.mode() != DatumDBParameter_Mode_READ);
  CHECK(batch_);
  string value;
  datum.SerializeToString(&value);
  batch_->Put(key, value);
}

void DatumLevelDB::Commit() {
  CHECK(param_.mode() != DatumDBParameter_Mode_READ);
  CHECK(batch_);
  leveldb::Status status = db_->Write(leveldb::WriteOptions(), batch_);
  CHECK(status.ok()) << "Failed to write batch to leveldb " << param_.source()
                     << std::endl << status.ToString();
  delete batch_;
  batch_ = new leveldb::WriteBatch();
}

}  // namespace caffe
