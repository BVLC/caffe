#include <glog/logging.h>
#include <stdint.h>

#include <algorithm>
#include <string>

#include "caffe/database_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using caffe::Database;
using caffe::Datum;
using caffe::BlobProto;
using std::max;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 3 || argc > 4) {
    LOG(ERROR) << "Usage: compute_image_mean input_db output_file"
               << " db_backend[leveldb or lmdb]";
    return 1;
  }

  std::string db_backend = "lmdb";
  if (argc == 4) {
    db_backend = std::string(argv[3]);
  }

  caffe::shared_ptr<Database> database = caffe::DatabaseFactory(db_backend);

  // Open db
  database->open(argv[1], Database::ReadOnly);

  Datum datum;
  BlobProto sum_blob;
  int count = 0;
  // load first datum
  Database::const_iterator iter = database->begin();
  const Database::buffer_t& first_blob = iter->value;
  datum.ParseFromArray(first_blob.data(), first_blob.size());
  iter = database->end();

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
  for (Database::const_iterator iter = database->begin();
      iter != database->end(); ++iter) {
    // just a dummy operation
    const Database::buffer_t& blob = iter->value;
    datum.ParseFromArray(blob.data(), blob.size());
    const std::string& data = datum.data();
    size_in_datum = std::max<int>(datum.data().size(),
        datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
    if (data.size() != 0) {
      CHECK_EQ(data.size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
      }
    } else {
      CHECK_EQ(datum.float_data_size(), size_in_datum);
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

  // Clean up
  database->close();
  return 0;
}
