#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/dataset_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using caffe::Dataset;
using caffe::Datum;
using std::pair;
using std::vector;


DEFINE_string(backend, "lmdb", "The backend for containing the images");
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

  std::string db_backend = FLAGS_backend;
  const int max_iter = FLAGS_max_iter;

  caffe::shared_ptr<Dataset<std::string, Datum> > dataset =
      caffe::DatasetFactory<std::string, Datum>(db_backend);

  // Open db
  CHECK(dataset->open(argv[1], Dataset<std::string, Datum>::ReadOnly));

  int count = 0;
  // load first datum
  Dataset<std::string, Datum>::const_iterator iter = dataset->begin();
  LOG(INFO) << "Starting Iteration";
  std::vector<pair<std::string, int> > keys_label;
  for (Dataset<std::string, Datum>::const_iterator iter = dataset->begin();
      iter != dataset->end(); ++iter) {
    std::string key = iter->key;
    Datum datum = iter->value;
    int label = datum.label();
    keys_label.push_back(std::make_pair(key, label));
    ++count;
    if (count % 10000 == 0) {
      LOG(INFO) << "Processed " << count << " items.";
    }
    if (max_iter > 0 && count >= max_iter) {
      break;
    }
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

  // Clean up
  dataset->close();
  return 0;
}
