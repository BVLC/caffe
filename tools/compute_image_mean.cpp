#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdint.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "caffe/datum_DB.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;


DEFINE_string(backend, "lmdb",
        "The backend {leveldb, lmdb, imagesdb} containing the images");
DEFINE_int32(max_num_images, 0,
        "Max number of images to process (0 means use all)");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
        " a leveldb/lmdb/imagesdb\n"
        "Usage:\n"
        "    compute_image_mean [FLAGS] INPUT_DB [OUTPUT_FILE]\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 2 || argc > 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
    return 1;
  }

  const int max_num_images = FLAGS_max_num_images;

  DatumDBParameter datumdb_param;
  datumdb_param.set_source(argv[1]);
  datumdb_param.set_loop(false);
  datumdb_param.set_backend(FLAGS_backend);
  shared_ptr<DatumDB> datumdb = DatumDB::GetDatumDB(datumdb_param);
  shared_ptr<DatumDB::Generator> datum_generator = datumdb->GetGenerator();

  BlobProto sum_blob;
  int count = 0;
  // load first datum
  Datum datum;
  CHECK(datum_generator->Current(&datum));

  if (DecodeDatum(&datum)) {
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
  while (datum_generator->Valid()) {
    Datum datum;
    CHECK(datum_generator->Current(&datum));
    DecodeDatum(&datum);

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
      LOG(INFO) << "Processed " << count << " files.";
    }
    if (max_num_images > 0 && count >= max_num_images) {
      break;
    }
    datum_generator->Next();
  }

  if (count % 10000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  if (argc == 3) {
    LOG(INFO) << "Write to " << argv[2];
    WriteProtoToBinaryFile(sum_blob, argv[2]);
  }
  const int channels = sum_blob.channels();
  const int dim = sum_blob.height() * sum_blob.width();
  std::vector<float> mean_values(channels, 0.0);
  LOG(INFO) << "Number of channels: " << channels;
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < dim; ++i) {
      mean_values[c] += sum_blob.data(dim * c + i);
    }
    LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
  }
  return 0;
}
