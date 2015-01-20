#include <string>

#include "leveldb/write_batch.h"

#include "caffe/datum_DB.hpp"

namespace caffe {

void DatumLevelDB::Open() {
  if (*is_opened_ == false) {
    LOG(INFO) << "Opening leveldb " << param_.source();
    leveldb::DB* db_temp;
    leveldb::Options options;
    options.block_size = 65536;
    options.write_buffer_size = 268435456;
    options.max_open_files = 100;
    switch (param_.mode()) {
    case DatumDBParameter_Mode_NEW:
      options.error_if_exists = true;
      options.create_if_missing = true;
      batch_.reset(new leveldb::WriteBatch());
      break;
    case DatumDBParameter_Mode_WRITE:
      options.error_if_exists = false;
      options.create_if_missing = true;
      batch_.reset(new leveldb::WriteBatch());
      break;
    case DatumDBParameter_Mode_READ:
      options.error_if_exists = false;
      options.create_if_missing = false;
      batch_.reset();
      break;
    default:
      LOG(FATAL) << "Unknown DB Mode " << param_.mode();
    }
    leveldb::Status status = leveldb::DB::Open(options,
                              param_.source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb " << param_.source()
                       << std::endl << status.ToString();
    db_.reset(db_temp);
    *is_opened_ = true;
  }
}

void DatumLevelDB::Close() {
  if (*is_opened_ && is_opened_.unique()) {
    LOG(INFO) << "Closing DatumLevelDB " << param_.source();
    batch_.reset();
    db_.reset();
  }
}

bool DatumLevelDB::Get(const string& key, Datum* datum) {
  const leveldb::Slice slice_key = key;
  string value;
  leveldb::Status status =
      db_->Get(leveldb::ReadOptions(), slice_key, &value);
  if (status.ok()) {
    datum->ParseFromString(value);
    return true;
  } else {
    LOG(ERROR) << status.ToString();
    return false;
  }
}

void DatumLevelDB::Put(const string& key, const Datum& datum) {
  CHECK(param_.mode() != DatumDBParameter_Mode_READ);
  CHECK_NOTNULL(batch_.get());
  string value;
  datum.SerializeToString(&value);
  batch_->Put(key, value);
}

void DatumLevelDB::Commit() {
  CHECK(param_.mode() != DatumDBParameter_Mode_READ);
  CHECK_NOTNULL(batch_.get());
  leveldb::Status status = db_->Write(leveldb::WriteOptions(), batch_.get());
  CHECK(status.ok()) << "Failed to write batch to leveldb " << param_.source()
                     << std::endl << status.ToString();
  batch_.reset(new leveldb::WriteBatch());
}

DatumDBCursor* DatumLevelDB::NewCursor() {
  CHECK_EQ(param_.mode(), DatumDBParameter_Mode_READ)
    << "Only DatumDB in Mode_READ can create NewCursor";
  CHECK(*is_opened_);
  LOG(INFO) << "Creating NewCursor for " << param_.source();
  leveldb::Iterator* iter = db_->NewIterator(leveldb::ReadOptions());
  return new DatumLevelDBCursor(param_, iter);
}

bool DatumLevelDBCursor::Valid() {
  return (iter_->Valid());
}

void DatumLevelDBCursor::SeekToFirst() {
  iter_->SeekToFirst();
}

void DatumLevelDBCursor::Next() {
  CHECK(Valid());
  iter_->Next();
  if (!iter_->Valid()) {
    if (param_.loop()) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Reached the end and looping.";
      SeekToFirst();
    } else {
      LOG(ERROR) << "Reached the end and not looping.";
    }
  }
}

string DatumLevelDBCursor::key() {
  CHECK(Valid());
  string key = iter_->key().ToString();
  return key;
}

Datum DatumLevelDBCursor::value() {
  CHECK(Valid());
  Datum datum;
  datum.ParseFromString(iter_->value().ToString());
  return datum;
}

REGISTER_DATUMDB_CLASS("leveldb", DatumLevelDB);

}  // namespace caffe
