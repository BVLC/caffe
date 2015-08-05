// This program converts a set of vector<float>'s to a lmdb/leveldb by storing
// them as Datum proto buffers.
// Usage:
//   convert_vector [FLAGS] LISTFILE DB_NAME
//
// where LISTFILE should be a list of files as well as the accompanying vector
// of floats, in the format as:
//   subfolder1/file1.JPEG 0.2 0.3 0.1 0.25 0.15
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of vectors");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of vectors to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_vector [FLAGS] LISTFILE DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_vector");
    return 1;
  }

  std::ifstream infile(argv[1]);
  std::vector<std::pair<std::string, std::vector<float> > > lines;
  std::string filename;

  std::string line;
  while (std::getline(infile, line)) {
    float vec;
    std::istringstream iss(line);
    iss >> filename;
    std::vector<float> vecs;
    while (iss >> vec) {
      vecs.push_back(vec);
    }
    lines.push_back(std::make_pair(filename, vecs));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  if (lines.size() < 1) {
    LOG(INFO) << "Read " << lines.size() << " vectors, aborting.";
    return 1;
  }
  LOG(INFO) << "A total of " << lines.size() << " vectors.";

  // Create new DB
  std::string dbname = argv[2];

  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(dbname, db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    // sequential
    int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
        lines[line_id].first.c_str());

    Datum datum;
    for (int i = 0; i < lines[line_id].second.size(); ++i) {
        datum.add_float_data(lines[line_id].second[i]);
    }
    datum.set_channels(lines[line_id].second.size());
    datum.set_height(1);
    datum.set_width(1);

    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(string(key_cstr, length), out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(ERROR) << "Processed " << count << " vectors.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(ERROR) << "Processed " << count << " vectors.";
  }
  return 0;
}

