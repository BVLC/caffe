// Copyright 2013 Yangqing Jia

#include <glog/logging.h>
#include <leveldb/db.h>

#include <string>

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = false;

  LOG(INFO) << "Opening leveldb " << argv[1];
  leveldb::Status status = leveldb::DB::Open(
      options, argv[1], &db);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[1];

  leveldb::ReadOptions read_options;
  read_options.fill_cache = false;
  int count = 0;
  leveldb::Iterator* it = db->NewIterator(read_options);
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    // just a dummy operation
    it->value().ToString();
    // LOG(ERROR) << it->key().ToString();
    if (++count % 10000 == 0) {
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }

  delete db;
  return 0;
}
