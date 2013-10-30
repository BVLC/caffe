// Copyright Yangqing Jia 2013
//
// This script converts the MNIST dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_mnist_data input_image_file input_label_file output_db_file
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/

#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <leveldb/db.h>

#include <stdint.h>
#include <iostream>
#include <fstream>
#include <string>

#include "caffe/proto/caffe.pb.h"

using std::string;


const int kCIFAR_SIZE=32;
const int kCIFAR_IMAGE_NBYTES=3072;
const int kCIFAR_BATCHSIZE=10000;
const int kCIFAR_TRAIN_BATCHES=5;

void read_image(std::ifstream& file, int* label, char* buffer) {
  char label_char;
  file.read(&label_char, 1);
  *label = label_char;
  file.read(buffer, kCIFAR_IMAGE_NBYTES);
  return;
}

void convert_dataset(const string& input_folder, const string& output_folder) {
  // Leveldb options
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  // Data buffer
  int label;
  char str_buffer[kCIFAR_IMAGE_NBYTES];
  string value;
  caffe::Datum datum;
  datum.set_channels(3);
  datum.set_height(kCIFAR_SIZE);
  datum.set_width(kCIFAR_SIZE);

  LOG(INFO) << "Writing Training data";
  leveldb::DB* train_db;
  leveldb::Status status;
  status = leveldb::DB::Open(options, output_folder + "/cifar-train-leveldb",
      &train_db);
  CHECK(status.ok()) << "Failed to open leveldb.";
  for (int fileid = 0; fileid < kCIFAR_TRAIN_BATCHES; ++fileid) {
    // Open files
    LOG(INFO) << "Training Batch " << fileid + 1;
    sprintf(str_buffer, "/data_batch_%d.bin", fileid + 1);
    std::ifstream data_file((input_folder + str_buffer).c_str(),
        std::ios::in | std::ios::binary);
    CHECK(data_file) << "Unable to open train file #" << fileid + 1;
    for (int itemid = 0; itemid < kCIFAR_BATCHSIZE; ++itemid) {
      read_image(data_file, &label, str_buffer);
      datum.set_label(label);
      datum.set_data(str_buffer, kCIFAR_IMAGE_NBYTES);
      datum.SerializeToString(&value);
      sprintf(str_buffer, "%05d", fileid * kCIFAR_BATCHSIZE + itemid);
      train_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
    }
  }

  LOG(INFO) << "Writing Testing data";
  leveldb::DB* test_db;
  CHECK(leveldb::DB::Open(options, output_folder + "/cifar-test-leveldb",
      &test_db).ok()) << "Failed to open leveldb.";
  // Open files
  std::ifstream data_file((input_folder + "/test_batch.bin").c_str(),
      std::ios::in | std::ios::binary);
  CHECK(data_file) << "Unable to open test file.";
  for (int itemid = 0; itemid < kCIFAR_BATCHSIZE; ++itemid) {
    read_image(data_file, &label, str_buffer);
    datum.set_label(label);
    datum.set_data(str_buffer, kCIFAR_IMAGE_NBYTES);
    datum.SerializeToString(&value);
    sprintf(str_buffer, "%05d", itemid);
    test_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
  }

  delete train_db;
  delete test_db;
}

int main (int argc, char** argv) {
  if (argc != 3) {
    printf("This script converts the CIFAR dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_cifar_data input_folder output_folder\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]));
  }
  return 0;
}
