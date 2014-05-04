// Copyright 2014 BVLC and contributors.

#include <glog/logging.h>
#include <lmdb.h>
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

  MDB_env* env;
  MDB_dbi dbi;
  MDB_val key, value;
  MDB_txn* txn;
  MDB_cursor* cursor;

  CHECK_EQ(mdb_env_create(&env), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(env, 1099511627776), MDB_SUCCESS); // 1TB
  CHECK_EQ(mdb_env_open(env, argv[1], MDB_RDONLY, 0664),
      MDB_SUCCESS) << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), MDB_SUCCESS)
      << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(txn, NULL, 0, &dbi), MDB_SUCCESS) << "mdb_open failed";
  CHECK_EQ(mdb_cursor_open(txn, dbi, &cursor), MDB_SUCCESS)
      << "mdb_cursor_open failed";

  LOG(INFO) << "Opening leveldb " << argv[1];
  CHECK_EQ(mdb_cursor_get(cursor, &key, &value, MDB_FIRST), MDB_SUCCESS);
  Datum datum;
  BlobProto sum_blob;
  int count = 0;
  datum.ParseFromArray(value.mv_data, value.mv_size);
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
  CHECK_EQ(mdb_cursor_get(cursor, &key, &value, MDB_FIRST), MDB_SUCCESS);
  do {
    // just a dummy operation
    datum.ParseFromArray(value.mv_data, value.mv_size);
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
  } while (mdb_cursor_get(cursor, &key, &value, MDB_NEXT) == MDB_SUCCESS);
  if (count % 10000 != 0) {
    LOG(ERROR) << "Processed " << count << " files.";
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  LOG(INFO) << "Write to " << argv[2];
  WriteProtoToBinaryFile(sum_blob, argv[2]);
  // Clean up the shit
  mdb_cursor_close(cursor);
  mdb_close(env, dbi);
  mdb_txn_abort(txn);
  mdb_env_close(env);
  return 0;
}
