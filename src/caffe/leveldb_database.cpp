#include <string>
#include <utility>

#include "caffe/leveldb_database.hpp"

namespace caffe {

bool LeveldbDatabase::open(const string& filename, Mode mode) {
  LOG(INFO) << "LevelDB: Open " << filename;

  leveldb::Options options;
  switch (mode) {
  case New:
    LOG(INFO) << " mode NEW";
    options.error_if_exists = true;
    options.create_if_missing = true;
    read_only_ = false;
    break;
  case ReadWrite:
    LOG(INFO) << " mode RW";
    options.error_if_exists = false;
    options.create_if_missing = true;
    read_only_ = false;
    break;
  case ReadOnly:
    LOG(INFO) << " mode RO";
    options.error_if_exists = false;
    options.create_if_missing = false;
    read_only_ = true;
    break;
  default:
    LOG(FATAL) << "unknown mode " << mode;
  }
  options.write_buffer_size = 268435456;
  options.max_open_files = 100;

  leveldb::DB* db;

  LOG(INFO) << "Opening leveldb " << filename;
  leveldb::Status status = leveldb::DB::Open(
      options, filename, &db);
  db_.reset(db);

  if (!status.ok()) {
    LOG(ERROR) << "Failed to open leveldb " << filename
        << ". Is it already existing?";
    return false;
  }

  batch_.reset(new leveldb::WriteBatch());
  return true;
}

bool LeveldbDatabase::put(const key_type& key, const value_type& value) {
  LOG(INFO) << "LevelDB: Put";

  if (read_only_) {
    LOG(ERROR) << "put can not be used on a database in ReadOnly mode";
    return false;
  }

  CHECK_NOTNULL(batch_.get());

  leveldb::Slice key_slice(key.data(), key.size());
  leveldb::Slice value_slice(value.data(), value.size());

  batch_->Put(key_slice, value_slice);

  return true;
}

bool LeveldbDatabase::get(const key_type& key, value_type* value) {
  LOG(INFO) << "LevelDB: Get";

  leveldb::Slice key_slice(key.data(), key.size());

  string value_string;
  leveldb::Status status =
      db_->Get(leveldb::ReadOptions(), key_slice, &value_string);

  if (!status.ok()) {
    LOG(ERROR) << "leveldb get failed";
    return false;
  }

  Database::value_type temp_value(value_string.data(),
      value_string.data() + value_string.size());
  value->swap(temp_value);

  return true;
}

bool LeveldbDatabase::commit() {
  LOG(INFO) << "LevelDB: Commit";

  if (read_only_) {
    LOG(ERROR) << "commit can not be used on a database in ReadOnly mode";
    return false;
  }

  CHECK_NOTNULL(db_.get());
  CHECK_NOTNULL(batch_.get());

  leveldb::Status status = db_->Write(leveldb::WriteOptions(), batch_.get());

  batch_.reset(new leveldb::WriteBatch());

  return status.ok();
}

void LeveldbDatabase::close() {
  LOG(INFO) << "LevelDB: Close";

  batch_.reset();
  db_.reset();
}

LeveldbDatabase::const_iterator LeveldbDatabase::begin() const {
  CHECK_NOTNULL(db_.get());
  shared_ptr<leveldb::Iterator> iter(db_->NewIterator(leveldb::ReadOptions()));
  iter->SeekToFirst();
  if (!iter->Valid()) {
    iter.reset();
  }
  shared_ptr<DatabaseState> state(new LeveldbState(db_, iter));
  return const_iterator(this, state);
}

LeveldbDatabase::const_iterator LeveldbDatabase::end() const {
  shared_ptr<leveldb::Iterator> iter;
  shared_ptr<DatabaseState> state(new LeveldbState(db_, iter));
  return const_iterator(this, state);
}

LeveldbDatabase::const_iterator LeveldbDatabase::cbegin() const {
  return begin();
}

LeveldbDatabase::const_iterator LeveldbDatabase::cend() const { return end(); }

bool LeveldbDatabase::equal(shared_ptr<DatabaseState> state1,
    shared_ptr<DatabaseState> state2) const {
  shared_ptr<LeveldbState> leveldb_state1 =
      boost::dynamic_pointer_cast<LeveldbState>(state1);

  CHECK_NOTNULL(leveldb_state1.get());

  shared_ptr<LeveldbState> leveldb_state2 =
      boost::dynamic_pointer_cast<LeveldbState>(state2);

  CHECK_NOTNULL(leveldb_state2.get());

  CHECK(!leveldb_state1->iter_ || leveldb_state1->iter_->Valid());
  CHECK(!leveldb_state2->iter_ || leveldb_state2->iter_->Valid());

  // The KV store doesn't really have any sort of ordering,
  // so while we can do a sequential scan over the collection,
  // we can't really use subranges.
  return !leveldb_state1->iter_ && !leveldb_state2->iter_;
}

void LeveldbDatabase::increment(shared_ptr<DatabaseState> state) const {
  shared_ptr<LeveldbState> leveldb_state =
      boost::dynamic_pointer_cast<LeveldbState>(state);

  CHECK_NOTNULL(leveldb_state.get());

  shared_ptr<leveldb::Iterator>& iter = leveldb_state->iter_;

  CHECK_NOTNULL(iter.get());
  CHECK(iter->Valid());

  iter->Next();
  if (!iter->Valid()) {
    iter.reset();
  }
}

Database::KV& LeveldbDatabase::dereference(
    shared_ptr<DatabaseState> state) const {
  shared_ptr<LeveldbState> leveldb_state =
      boost::dynamic_pointer_cast<LeveldbState>(state);

  CHECK_NOTNULL(leveldb_state.get());

  shared_ptr<leveldb::Iterator>& iter = leveldb_state->iter_;

  CHECK_NOTNULL(iter.get());

  CHECK(iter->Valid());

  Database::key_type temp_key(key_type(iter->key().data(),
      iter->key().data() + iter->key().size()));

  Database::value_type temp_value(value_type(iter->value().data(),
      iter->value().data() + iter->value().size()));

  leveldb_state->kv_pair_.key.swap(temp_key);
  leveldb_state->kv_pair_.value.swap(temp_value);
  return leveldb_state->kv_pair_;
}

}  // namespace caffe
