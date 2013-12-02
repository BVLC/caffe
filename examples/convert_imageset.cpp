// Copyright 2013 Yangqing Jia
// This program converts a set of images to a leveldb by storing them as Datum
// proto buffers.
// Usage:
//    convert_dataset ROOTFOLDER LISTFILE DB_NAME [0/1]
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....
// if the last argument is 1, a random shuffle will be carried out before we
// process the file lines.
// You are responsible for shuffling the files yourself.

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
using std::pair;
using std::string;
using std::stringstream;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 4) {
    LOG(ERROR) << "Usage: convert_imageset ROOTFOLDER LISTFILE DB_NAME [0/1]";
    return 0;
  }
  std::ifstream infile(argv[2]);
  std::vector<std::pair<string, int> > lines;
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  if (argc == 5 && argv[4][0] == '1') {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    std::random_shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  LOG(INFO) << "Opening leveldb " << argv[3];
  leveldb::Status status = leveldb::DB::Open(
      options, argv[3], &db);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[3];

  string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  char key_cstr[100];
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    if (!ReadImageToDatum(root_folder + lines[line_id].first, lines[line_id].second,
        &datum)) {
      continue;
    };
    // sequential
    sprintf(key_cstr, "%08d_%s", line_id, lines[line_id].first.c_str());
    string value;
    // get the value
    datum.SerializeToString(&value);
    batch->Put(string(key_cstr), value);
    if (++count % 1000 == 0) {
      db->Write(leveldb::WriteOptions(), batch);
      LOG(ERROR) << "Processed " << count << " files.";
      delete batch;
      batch = new leveldb::WriteBatch();
    }
  }

  delete db;
  return 0;
}
