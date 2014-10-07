#include <string>
#include <utility>

#include "caffe/leveldb_database.hpp"

namespace caffe {

void LeveldbDatabase::open(const string& filename, Mode mode) {
  LOG(INFO) << "LevelDB: Open " << filename;

  leveldb::Options options;
  switch (mode) {
  case New:
    LOG(INFO) << " mode NEW";
    options.error_if_exists = true;
    options.create_if_missing = true;
    break;
  case ReadWrite:
    LOG(INFO) << " mode RW";
    options.error_if_exists = false;
    options.create_if_missing = true;
    break;
  case ReadOnly:
    LOG(INFO) << " mode RO";
    options.error_if_exists = false;
    options.create_if_missing = false;
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
  CHECK(status.ok()) << "Failed to open leveldb " << filename
      << ". Is it already existing?";
  batch_.reset(new leveldb::WriteBatch());
}

void LeveldbDatabase::put(const string& key, const string& value) {
  LOG(INFO) << "LevelDB: Put " << key;

  CHECK_NOTNULL(batch_.get());
  batch_->Put(key, value);
}

void LeveldbDatabase::commit() {
  LOG(INFO) << "LevelDB: Commit";

  CHECK_NOTNULL(db_.get());
  CHECK_NOTNULL(batch_.get());

  db_->Write(leveldb::WriteOptions(), batch_.get());
  batch_.reset(new leveldb::WriteBatch());
}

void LeveldbDatabase::close() {
  LOG(INFO) << "LevelDB: Close";

  if (batch_ && db_) {
    this->commit();
  }
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
  shared_ptr<DatabaseState> state(new LeveldbState(iter));
  return const_iterator(this, state);
}

LeveldbDatabase::const_iterator LeveldbDatabase::end() const {
  shared_ptr<leveldb::Iterator> iter;
  shared_ptr<DatabaseState> state(new LeveldbState(iter));
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

pair<string, string>& LeveldbDatabase::dereference(
    shared_ptr<DatabaseState> state) const {
  shared_ptr<LeveldbState> leveldb_state =
      boost::dynamic_pointer_cast<LeveldbState>(state);

  CHECK_NOTNULL(leveldb_state.get());

  shared_ptr<leveldb::Iterator>& iter = leveldb_state->iter_;

  CHECK_NOTNULL(iter.get());

  CHECK(iter->Valid());

  leveldb_state->kv_pair_ = make_pair(iter->key().ToString(),
      iter->value().ToString());
  return leveldb_state->kv_pair_;
}

}  // namespace caffe
