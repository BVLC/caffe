#include "caffe/util/db.hpp"

#include <sys/stat.h>
#include <string>

namespace caffe { namespace db {

const size_t LMDB_MAP_SIZE = 1099511627776;  // 1 TB

void LevelDB::Open(const string& source, Mode mode) {
  leveldb::Options options;
  options.block_size = 65536;
  options.write_buffer_size = 268435456;
  options.max_open_files = 100;
  options.error_if_exists = mode == NEW;
  options.create_if_missing = mode != READ;
  leveldb::Status status = leveldb::DB::Open(options, source, &db_);
  CHECK(status.ok()) << "Failed to open leveldb " << source
                     << std::endl << status.ToString();
  LOG(INFO) << "Opened leveldb " << source;
}

void LMDB::Open(const string& source, Mode mode) {
  MDB_CHECK(mdb_env_create(&mdb_env_));
  MDB_CHECK(mdb_env_set_mapsize(mdb_env_, LMDB_MAP_SIZE));
  if (mode == NEW) {
    CHECK_EQ(mkdir(source.c_str(), 0744), 0) << "mkdir " << source << "failed";
  }
  int flags = 0;
  if (mode == READ) {
    flags = MDB_RDONLY | MDB_NOTLS;
  }
  MDB_CHECK(mdb_env_open(mdb_env_, source.c_str(), flags, 0664));
  LOG(INFO) << "Opened lmdb " << source;
}

LMDBCursor* LMDB::NewCursor() {
  MDB_txn* mdb_txn;
  MDB_cursor* mdb_cursor;
  MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn));
  MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_));
  MDB_CHECK(mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor));
  return new LMDBCursor(mdb_txn, mdb_cursor);
}

LMDBTransaction* LMDB::NewTransaction() {
  MDB_txn* mdb_txn;
  MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, 0, &mdb_txn));
  MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_));
  return new LMDBTransaction(&mdb_dbi_, mdb_txn);
}

void LMDBTransaction::Put(const string& key, const string& value) {
  MDB_val mdb_key, mdb_value;
  mdb_key.mv_data = const_cast<char*>(key.data());
  mdb_key.mv_size = key.size();
  mdb_value.mv_data = const_cast<char*>(value.data());
  mdb_value.mv_size = value.size();
  MDB_CHECK(mdb_put(mdb_txn_, *mdb_dbi_, &mdb_key, &mdb_value, 0));
}


void DatumFileCursor::SeekToFirst() {
    if (in_ && in_->is_open()) {
      in_->close();
    }
    LOG(INFO) << "reset ifstream " << path_;
    in_ = new std::ifstream(path_.c_str(),
            std::ifstream::in|std::ifstream::binary);
    Next();
  }

void DatumFileCursor::Next() {
  valid_ = false;
  CHECK(in_->is_open()) << "file is not open!" << path_;

  uint32_t record_size = 0, key_size = 0, value_size = 0;
  in_->read(reinterpret_cast<char*>(&record_size), sizeof record_size);
  if (in_->gcount() != (sizeof record_size) || record_size > MAX_BUF) {
    CHECK(in_->eof() && record_size <= MAX_BUF)
      <<"record_size read error: gcount\t"
      << in_->gcount() << "\trecord_size\t" << record_size;
    return;
  }

  in_->read(reinterpret_cast<char*>(&key_size), sizeof key_size);
  CHECK(in_->gcount() == sizeof key_size && key_size <= MAX_BUF)
    << "key_size read error: gcount\t"
    << in_->gcount() << "\tkey_size\t" << key_size;

  key_.resize(key_size);
  in_->read(&key_[0], key_size);
  CHECK(in_->gcount() == key_size)
    << "key read error: gcount\t"
    << in_->gcount() << "\tkey_size\t" << key_size;

  in_->read(reinterpret_cast<char*>(&value_size), sizeof value_size);
  CHECK(in_->gcount() == sizeof value_size && value_size <= MAX_BUF)
    << "value_size read error: gcount\t"
    << in_->gcount() << "\tvalue_size\t" << value_size;

  value_.resize(value_size);
  in_->read(&value_[0], value_size);
  CHECK(in_->gcount() == value_size)
    << "value read error: gcount\t"
    << in_->gcount() << "\tvalue_size\t" << value_size;

  valid_ = true;
}

void DatumFileTransaction::Put(const string& key, const string& value) {
  try {
    uint32_t key_size = key.size(), value_size = value.size();
    uint32_t record_size = key_size + value_size
        + sizeof key_size + sizeof value_size;
    out_->write(reinterpret_cast<char*>(&record_size), sizeof record_size);
    out_->write(reinterpret_cast<char*>(&key_size), sizeof key_size);
    out_->write(key.data(), key_size);
    out_->write(reinterpret_cast<char*>(&value_size), sizeof value_size);
    out_->write(value.data(), value_size);
  } catch(std::ios_base::failure& e) {
    LOG(FATAL) << "Exception: "
        << e.what() << " rdstate: " << out_->rdstate() << '\n';
  }
}

Transaction* DatumFileDB::NewTransaction() {
  if (!this->out_) {
    out_ = new std::ofstream();
    out_->open(this->path_.c_str(),
            std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
    out_->exceptions(out_->exceptions() | std::ios::failbit);
    LOG(INFO) << "Output created: " << path_ << std::endl;
  }
  return new DatumFileTransaction(this->out_);
}

DB* GetDB(DataParameter::DB backend) {
  switch (backend) {
  case DataParameter_DB_LEVELDB:
    return new LevelDB();
  case DataParameter_DB_LMDB:
    return new LMDB();
  case DataParameter_DB_DATUMFILE:
    return new DatumFileDB();
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

DB* GetDB(const string& backend) {
  if (backend == "leveldb") {
    return new LevelDB();
  } else if (backend == "lmdb") {
    return new LMDB();
  } else if (backend == "datumfile") {
    return new DatumFileDB();
  } else {
    LOG(FATAL) << "Unknown database backend";
  }
}

}  // namespace db
}  // namespace caffe
