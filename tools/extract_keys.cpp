#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/datum_DB.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::pair;
using std::vector;


DEFINE_string(backend, "lmdb",
    "The backend {lmdb, leveldb, imagesdb} containing the images");
DEFINE_int32(max_iter, 0, "Max number of iterations (0 means extract all)");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Extract keys and labels from a leveldb/lmdb \n"
        "Usage:\n"
        "    extract_keys [FLAGS] INPUT_DB [OUTPUT_FILE]\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 2 || argc > 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/extract_keys");
    return 1;
  }

  DatumDBParameter datumdb_param;
  datumdb_param.set_source(argv[1]);
  datumdb_param.set_loop(false);
  datumdb_param.set_backend(FLAGS_backend);
  // Usefull for imagesdb ignored for the rest while reading
  datumdb_param.set_encode_images(true);

  shared_ptr<DatumDB> datumdb(DatumDBRegistry::GetDatumDB(datumdb_param));
  shared_ptr<DatumDBCursor> datum_cursor(datumdb->NewCursor());

  const int max_iter = FLAGS_max_iter;

  int count = 0;
  // load first datum
  LOG(INFO) << "Starting Iteration";
  vector<pair<string, int> > keys_label;
  while (datum_cursor->Valid()) {
    string key = datum_cursor->key();
    Datum datum = datum_cursor->value();
    int label = datum.label();
    keys_label.push_back(std::make_pair(key, label));
    ++count;
    if (count % 10000 == 0) {
      LOG(INFO) << "Processed " << count << " items.";
    }
    if (max_iter > 0 && count >= max_iter) {
      break;
    }
    datum_cursor->Next();
  }

  if (count % 10000 != 0) {
    LOG(INFO) << "Processed " << count << " items.";
  }

  // Write to disk
  if (argc == 3) {
    LOG(INFO) << "Writting to " << argv[2];
    std::ofstream outfile(argv[2]);
    for (int i = 0; i < keys_label.size(); ++i) {
      outfile << keys_label[i].first << " " << keys_label[i].second
        << std::endl;
    }
    outfile.close();
  } else {
    // Write to cout
    for (int i = 0; i < keys_label.size(); ++i) {
      std::cout << keys_label[i].first << " " << keys_label[i].second
        << std::endl;
    }
  }
  return 0;
}
