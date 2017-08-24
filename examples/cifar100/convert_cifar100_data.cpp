//
// This script converts the CIFAR100 dataset to the lmdb/leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html
// Seek more information : Coderx7@Gmail.com

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

using caffe::Datum;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;

// the spatial size e.i height and width of images in cifar10/100 dataset
const int kCIFARSize = 32;
// the size of an image in cifar10/100 dataset (32*32*3=3072)
const int kCIFARImageNBytes = 3072;
// the number of images in a batch file
const int kCIFARBatchSize = 10000;
// in cifar100, there is only one batch of 50K training images
const int kCIFARTrainBatches = 1;

/*
As it is explained in the main site:
The binary version of the CIFAR-100 is just like the binary version of the
CIFAR-10,except that each image has two label bytes (coarse and fine) and 
3072 pixel bytes,so the binary files look like this:
<1 x coarse label><1 x fine label><3072 x pixel>
...
<1 x coarse label><1 x fine label><3072 x pixel>
which shows the layout in-which the data(our labels, and the image file) 
are layed out, which again says the first byte is a coarse label, the second
one is a fine label and the next 3072 bytes are the actual image
and this pattern repeats until the whole dataset is finished.
*/
void read_image(std::ifstream* file,
int* coarse_label, int* fine_label, char* buffer) {
  char clabel_char, flabel_char;
  file->read(&clabel_char, 1);  // read the first byte into coarse label
  file->read(&flabel_char, 1);  // read the second byte into fine label
  *coarse_label = clabel_char;
  *fine_label = flabel_char;
  file->read(buffer, kCIFARImageNBytes);  // read the next 3072 bytes
  return;
}

void convert_dataset(const string& input_folder,
  const string& output_folder,  const string& db_type) {
  scoped_ptr<db::DB> train_db(db::GetDB(db_type));
  train_db->Open(output_folder + "/cifar100_train_" + db_type, db::NEW);
  scoped_ptr<db::Transaction> txn(train_db->NewTransaction());
  // Data buffer
  int coarse_label, fine_label;
  char str_buffer[kCIFARImageNBytes];
  Datum datum;
  datum.set_channels(3);
  datum.set_height(kCIFARSize);
  datum.set_width(kCIFARSize);

  LOG(INFO) << "Writing Training data";
  // Open files
  LOG(INFO) << "Training Batch 1/1";
  string batchFileName = input_folder + "/train.bin";
  std::ifstream train_data_file(batchFileName.c_str(),
  std::ios::in | std::ios::binary);
  CHECK(train_data_file) << "Unable to open train file ";
  // we have 50,000 images for training
  int training_batch_size = kCIFARBatchSize * 5;
  for (int itemid = 0; itemid < training_batch_size; ++itemid) {
    read_image(&train_data_file, &coarse_label,
    &fine_label, str_buffer);
    datum.set_label(coarse_label);
    datum.set_label(fine_label);
    datum.set_data(str_buffer, kCIFARImageNBytes);
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(caffe::format_int(itemid, 5), out);
  }
  txn->Commit();
  train_db->Close();

  LOG(INFO) << "Writing Testing data";
  scoped_ptr<db::DB> test_db(db::GetDB(db_type));
  test_db->Open(output_folder + "/cifar100_test_" + db_type, db::NEW);
  txn.reset(test_db->NewTransaction());
  // Open files
  std::ifstream test_data_file((input_folder + "/test.bin").c_str(),
      std::ios::in | std::ios::binary);
  CHECK(test_data_file) << "Unable to open test file.";
  for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid)  {
    read_image(&test_data_file, &coarse_label, &fine_label, str_buffer);
    datum.set_label(coarse_label);
    datum.set_label(fine_label);
    datum.set_data(str_buffer, kCIFARImageNBytes);
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(caffe::format_int(itemid, 5), out);
  }
  txn->Commit();
  test_db->Close();
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("This script converts the CIFAR100 dataset to the lmdb/leveldb\n"
           " format used by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_cifar_data input_folder output_folder db_type\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR100 dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]), string(argv[3]));
  }
  return 0;
}
