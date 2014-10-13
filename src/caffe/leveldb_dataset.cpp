#include <string>
#include <utility>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/leveldb_dataset.hpp"

namespace caffe {

template <typename K, typename V>
bool LeveldbDataset<K, V>::open(const string& filename, Mode mode) {
  DLOG(INFO) << "LevelDB: Open " << filename;

  leveldb::Options options;
  switch (mode) {
  case Base::New:
    DLOG(INFO) << " mode NEW";
    options.error_if_exists = true;
    options.create_if_missing = true;
    read_only_ = false;
    break;
  case Base::ReadWrite:
    DLOG(INFO) << " mode RW";
    options.error_if_exists = false;
    options.create_if_missing = true;
    read_only_ = false;
    break;
  case Base::ReadOnly:
    DLOG(INFO) << " mode RO";
    options.error_if_exists = false;
    options.create_if_missing = false;
    read_only_ = true;
    break;
  default:
    DLOG(FATAL) << "unknown mode " << mode;
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

template <typename K, typename V>
bool LeveldbDataset<K, V>::put(const K& key, const V& value) {
  DLOG(INFO) << "LevelDB: Put";

  if (read_only_) {
    LOG(ERROR) << "put can not be used on a dataset in ReadOnly mode";
    return false;
  }

  CHECK_NOTNULL(batch_.get());

  string serialized_key;
  if (!Base::serialize(key, &serialized_key)) {
    return false;
  }

  string serialized_value;
  if (!Base::serialize(value, &serialized_value)) {
    return false;
  }

  batch_->Put(serialized_key, serialized_value);

  return true;
}

template <typename K, typename V>
bool LeveldbDataset<K, V>::get(const K& key, V* value) {
  DLOG(INFO) << "LevelDB: Get";

  string serialized_key;
  if (!Base::serialize(key, &serialized_key)) {
    return false;
  }

  string serialized_value;
  leveldb::Status status =
      db_->Get(leveldb::ReadOptions(), serialized_key, &serialized_value);

  if (!status.ok()) {
    LOG(ERROR) << "leveldb get failed";
    return false;
  }

  if (!Base::deserialize(serialized_value, value)) {
    return false;
  }

  return true;
}

template <typename K, typename V>
bool LeveldbDataset<K, V>::commit() {
  DLOG(INFO) << "LevelDB: Commit";

  if (read_only_) {
    LOG(ERROR) << "commit can not be used on a dataset in ReadOnly mode";
    return false;
  }

  CHECK_NOTNULL(db_.get());
  CHECK_NOTNULL(batch_.get());

  leveldb::Status status = db_->Write(leveldb::WriteOptions(), batch_.get());

  batch_.reset(new leveldb::WriteBatch());

  return status.ok();
}

template <typename K, typename V>
void LeveldbDataset<K, V>::close() {
  DLOG(INFO) << "LevelDB: Close";

  batch_.reset();
  db_.reset();
}

template <typename K, typename V>
void LeveldbDataset<K, V>::keys(vector<K>* keys) {
  DLOG(INFO) << "LevelDB: Keys";

  keys->clear();
  for (const_iterator iter = begin(); iter != end(); ++iter) {
    keys->push_back(iter->key);
  }
}

template <typename K, typename V>
typename LeveldbDataset<K, V>::const_iterator
    LeveldbDataset<K, V>::begin() const {
  CHECK_NOTNULL(db_.get());
  shared_ptr<leveldb::Iterator> iter(db_->NewIterator(leveldb::ReadOptions()));
  iter->SeekToFirst();
  if (!iter->Valid()) {
    iter.reset();
  }

  shared_ptr<DatasetState> state;
  if (iter) {
    state.reset(new LeveldbState(db_, iter));
  }
  return const_iterator(this, state);
}

template <typename K, typename V>
typename LeveldbDataset<K, V>::const_iterator
    LeveldbDataset<K, V>::end() const {
  shared_ptr<DatasetState> state;
  return const_iterator(this, state);
}

template <typename K, typename V>
typename LeveldbDataset<K, V>::const_iterator
    LeveldbDataset<K, V>::cbegin() const {
  return begin();
}

template <typename K, typename V>
typename LeveldbDataset<K, V>::const_iterator
    LeveldbDataset<K, V>::cend() const { return end(); }

template <typename K, typename V>
bool LeveldbDataset<K, V>::equal(shared_ptr<DatasetState> state1,
    shared_ptr<DatasetState> state2) const {
  shared_ptr<LeveldbState> leveldb_state1 =
      boost::dynamic_pointer_cast<LeveldbState>(state1);

  shared_ptr<LeveldbState> leveldb_state2 =
      boost::dynamic_pointer_cast<LeveldbState>(state2);

  // The KV store doesn't really have any sort of ordering,
  // so while we can do a sequential scan over the collection,
  // we can't really use subranges.
  return !leveldb_state1 && !leveldb_state2;
}

template <typename K, typename V>
void LeveldbDataset<K, V>::increment(shared_ptr<DatasetState>* state) const {
  shared_ptr<LeveldbState> leveldb_state =
      boost::dynamic_pointer_cast<LeveldbState>(*state);

  CHECK_NOTNULL(leveldb_state.get());

  shared_ptr<leveldb::Iterator>& iter = leveldb_state->iter_;

  CHECK_NOTNULL(iter.get());
  CHECK(iter->Valid());

  iter->Next();
  if (!iter->Valid()) {
    state->reset();
  }
}

template <typename K, typename V>
typename Dataset<K, V>::KV& LeveldbDataset<K, V>::dereference(
    shared_ptr<DatasetState> state) const {
  shared_ptr<LeveldbState> leveldb_state =
      boost::dynamic_pointer_cast<LeveldbState>(state);

  CHECK_NOTNULL(leveldb_state.get());

  shared_ptr<leveldb::Iterator>& iter = leveldb_state->iter_;

  CHECK_NOTNULL(iter.get());

  CHECK(iter->Valid());

  const leveldb::Slice& key = iter->key();
  const leveldb::Slice& value = iter->value();
  CHECK(Base::deserialize(key.data(), key.size(),
      &leveldb_state->kv_pair_.key));
  CHECK(Base::deserialize(value.data(), value.size(),
      &leveldb_state->kv_pair_.value));

  return leveldb_state->kv_pair_;
}

INSTANTIATE_DATASET(LeveldbDataset);

}  // namespace caffe
