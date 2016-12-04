#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
        "The backend {leveldb, lmdb} containing the images");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifdef USE_OPENCV
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Compute the sd_image of a set of images given by"
        " a leveldb/lmdb\n"
        "Usage:\n"
        "    compute_image_sd [FLAGS] INPUT_DB MEAN_FILE OUTPUT_FILE\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_sd");
    return 1;
  }

  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[1], db::READ);
  scoped_ptr<db::Cursor> cursor(db->NewCursor());

  // Load mean file
  BlobProto mean_blob;
  ReadProtoFromBinaryFileOrDie(argv[2], &mean_blob);

  BlobProto sum_blob;
  int count = 0;
  // load first datum
  Datum datum;
  datum.ParseFromString(cursor->value());

  if (DecodeDatumNative(&datum)) {
    LOG(INFO) << "Decoding Datum";
  }

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
  while (cursor->valid()) {
    Datum datum;
    datum.ParseFromString(cursor->value());
    DecodeDatumNative(&datum);

    const std::string& data = datum.data();
    size_in_datum = std::max<int>(datum.data().size(),
        datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
    if (data.size() != 0) {
      CHECK_EQ(data.size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        float val = (uint8_t)data[i];
        val -= mean_blob.data(i);
        sum_blob.set_data(i, sum_blob.data(i) + val*val);
      }
    } else {
      CHECK_EQ(datum.float_data_size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        float val = static_cast<float>(datum.float_data(i));
        val -= mean_blob.data(i);
        sum_blob.set_data(i, sum_blob.data(i) + val*val);
      }
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
    cursor->Next();
  }

  if (count % 10000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }

  // Compute per channel standard deviation
  const int channels = sum_blob.channels();
  const int dim = sum_blob.height() * sum_blob.width();
  std::vector<float> sd_values(channels, 0.0);
  LOG(INFO) << "Number of channels: " << channels;
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < dim; ++i) {
      sd_values[c] += sum_blob.data(dim * c + i);
    }
	sd_values[c] = static_cast<float>((double)sd_values[c] / (double)count * (double)dim);
    LOG(INFO) << "sd_value channel [" << c << "]:" << std::sqrt(sd_values[c]);
  }

  // Compute per element standard deviation
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    float sd = sum_blob.data(i) / count;
    sd = std::sqrt(sd);
    sum_blob.set_data(i, sd);
  }
  // Write to disk
  if (argc == 4) {
	LOG(INFO) << "Write to " << argv[3];
	WriteProtoToBinaryFile(sum_blob, argv[3]);
  }  
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
