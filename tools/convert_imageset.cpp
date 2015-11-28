// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======

#include <gflags/gflags.h>
#include <glog/logging.h>
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
#include "caffe/dataset_factory.hpp"
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
<<<<<<< HEAD
using boost::scoped_ptr;
=======
<<<<<<< HEAD
<<<<<<< HEAD
using boost::scoped_ptr;
=======
>>>>>>> origin/BVLC/parallel
=======
using boost::scoped_ptr;
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
<<<<<<< HEAD
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
=======
<<<<<<< HEAD
<<<<<<< HEAD
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
=======
DEFINE_string(backend, "lmdb", "The backend for storing the result");
>>>>>>> origin/BVLC/parallel
=======
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
<<<<<<< HEAD
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");
=======
<<<<<<< HEAD
<<<<<<< HEAD
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");
=======
>>>>>>> origin/BVLC/parallel
=======
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
<<<<<<< HEAD
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
=======
<<<<<<< HEAD
<<<<<<< HEAD
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
=======
>>>>>>> origin/BVLC/parallel
=======
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

<<<<<<< HEAD
  if (argc < 4) {
=======
<<<<<<< HEAD
<<<<<<< HEAD
  if (argc < 4) {
=======
  if (argc != 4) {
>>>>>>> origin/BVLC/parallel
=======
  if (argc < 4) {
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
<<<<<<< HEAD
  const string encode_type = FLAGS_encode_type;
=======
<<<<<<< HEAD
<<<<<<< HEAD
  const string encode_type = FLAGS_encode_type;
=======
>>>>>>> origin/BVLC/parallel
=======
  const string encode_type = FLAGS_encode_type;
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, int> > lines;
  std::string filename;
  int label;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);
<<<<<<< HEAD

=======
<<<<<<< HEAD

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());
=======
  const std::string& db_backend = FLAGS_backend;
  const char* db_path = argv[3];

  if (encoded) {
    CHECK_EQ(FLAGS_resize_height, 0) << "With encoded don't resize images";
    CHECK_EQ(FLAGS_resize_width, 0) << "With encoded don't resize images";
    CHECK(!check_size) << "With encoded cannot check_size";
  }

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Open new db
  shared_ptr<Dataset<string, Datum> > dataset =
      DatasetFactory<string, Datum>(db_backend);

  // Open db
  CHECK(dataset->open(db_path, Dataset<string, Datum>::New));
>>>>>>> origin/BVLC/parallel
=======

>>>>>>> pod-caffe-pod.hpp-merge
  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    status = ReadImageToDatum(root_folder + lines[line_id].first,
        lines[line_id].second, resize_height, resize_width, is_color,
        enc, &datum);
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
    if (encoded) {
      status = ReadFileToDatum(root_folder + lines[line_id].first,
        lines[line_id].second, &datum);
    } else {
      status = ReadImageToDatum(root_folder + lines[line_id].first,
          lines[line_id].second, resize_height, resize_width, is_color, &datum);
    }
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
<<<<<<< HEAD
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
=======
<<<<<<< HEAD
<<<<<<< HEAD
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
=======
    int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
        lines[line_id].first.c_str());

    // Put in db
    CHECK(dataset->put(string(key_cstr, length), datum));

    if (++count % 1000 == 0) {
      // Commit txn
      CHECK(dataset->commit());
      LOG(ERROR) << "Processed " << count << " files.";
>>>>>>> origin/BVLC/parallel
=======
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
>>>>>>> pod-caffe-pod.hpp-merge
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
<<<<<<< HEAD
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
=======
<<<<<<< HEAD
<<<<<<< HEAD
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
=======
    CHECK(dataset->commit());
    LOG(ERROR) << "Processed " << count << " files.";
  }
  dataset->close();
>>>>>>> origin/BVLC/parallel
=======
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  return 0;
}
