// Copyright 2014 BVLC and contributors.

#include <glog/logging.h>
#include <leveldb/db.h>
#include <stdint.h>

#include <algorithm>
#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using caffe::Datum;
using caffe::BlobProto;
using std::max;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    LOG(ERROR) << "Usage: compute_image_mean input_leveldb output_file";
    return 1;
  }

  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = false;

  LOG(INFO) << "Opening leveldb " << argv[1];
  leveldb::Status status = leveldb::DB::Open(
      options, argv[1], &db);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[1];

  leveldb::ReadOptions read_options;
  read_options.fill_cache = false;
  leveldb::Iterator* it = db->NewIterator(read_options);
  it->SeekToFirst();
  Datum datum;
  BlobProto sum_blob;
  int count = 0;
  datum.ParseFromString(it->value().ToString());
  sum_blob.set_num(1);
  sum_blob.set_channels(datum.channels());
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());
  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }
  LOG(INFO) << "Starting Iteration";
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    // just a dummy operation
    datum.ParseFromString(it->value().ToString());
    const string& data = datum.data();
    size_in_datum = std::max<int>(datum.data().size(), datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
    if (data.size() != 0) {
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
      }
    } else {
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) +
            static_cast<float>(datum.float_data(i)));
      }
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  if (count % 10000 != 0) {
    LOG(ERROR) << "Processed " << count << " files.";
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  LOG(INFO) << "Write to " << argv[2];
  WriteProtoToBinaryFile(sum_blob, argv[2]);

  delete db;
  return 0;
}
