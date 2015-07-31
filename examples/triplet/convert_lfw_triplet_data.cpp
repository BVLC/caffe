// Usage:
//    convert_mnist_data input_image_file input_label_file output_db_file
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "stdint.h"

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void read_image(std::ifstream* image_file, std::ifstream* label_file,
        uint32_t index, uint32_t rows, uint32_t cols,
        char* pixels, char* label_temp, signed char* label) {
  image_file->seekg(index * rows * cols + 16);
  image_file->read(pixels, rows * cols);
  label_file->seekg(index * 4 + 8);
  label_file->read(label_temp, 4);
  for (int i = 0; i < 4; i++)
    *(label+i) = (signed char)*(label_temp+i);
}

void convert_dataset(const char* image_filename, const char* label_filename,
        const char* db_filename) {
  // Open files
  std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
  std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
  CHECK(image_file) << "Unable to open file " << image_filename;
  CHECK(label_file) << "Unable to open file " << label_filename;
  // Read the magic and the meta data
  uint32_t magic;
  uint32_t num_items;
  uint32_t num_labels;
  uint32_t rows;
  uint32_t cols;

  image_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
  label_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2050) << "Incorrect label file magic.";
  image_file.read(reinterpret_cast<char*>(&num_items), 4);
  num_items = swap_endian(num_items);
  label_file.read(reinterpret_cast<char*>(&num_labels), 4);
  num_labels = swap_endian(num_labels);
  CHECK_EQ(num_items, num_labels);
  image_file.read(reinterpret_cast<char*>(&rows), 4);
  rows = swap_endian(rows);
  image_file.read(reinterpret_cast<char*>(&cols), 4);
  cols = swap_endian(cols);

  // Open leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_filename
      << ". Is it already existing?";

  char* label_temp = new char[4];  // label for unsigned char*
  signed char* label_i = new signed char[4];  // label for triplet
  signed char* label_j = new signed char[4];
  signed char* label_k = new signed char[4];
  signed char* label_l = new signed char[4];  // label for pair wise
  signed char* label_m = new signed char[4];
  char* pixels = new char[5 * rows * cols];
  const int kMaxKeyLength = 10;
  char key[kMaxKeyLength];
  std::string value;

  caffe::Datum datum;
  datum.set_channels(5);  // one channel for each image in the triplet and pair
  datum.set_height(rows);
  datum.set_width(cols);
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  for (unsigned int itemid = 0; itemid < 10 * num_items; ++itemid) {\
    int i = caffe::caffe_rng_rand() % num_items;  // pick triplet groups
    int j = caffe::caffe_rng_rand() % num_items;
    int k = caffe::caffe_rng_rand() % num_items;
    int l = caffe::caffe_rng_rand() % num_items;  // pick pair wise groups
    int m = caffe::caffe_rng_rand() % num_items;
    read_image(&image_file, &label_file, i, rows, cols,  // read triplet
        pixels, label_temp, label_i);
    read_image(&image_file, &label_file, j, rows, cols,
        pixels + (rows * cols), label_temp, label_j);
    read_image(&image_file, &label_file, k, rows, cols,
        pixels + (2 * rows * cols), label_temp, label_k);
    read_image(&image_file, &label_file, l, rows, cols,  // read pair wise
        pixels + (3 * rows * cols), label_temp, label_l);
    read_image(&image_file, &label_file, m, rows, cols,
        pixels + (4 * rows * cols), label_temp, label_m);

    datum.set_data(pixels, 5*rows*cols);  // set data
    bool triplet_class_pass = false;
    bool triplet_class_same = false;
    bool triplet_pose_pass = false;
    bool pair_class_pass = false;

    int ij_diff_x = static_cast<int>(*(label_i+1)-*(label_j+1));
    int ij_diff_y = static_cast<int>(*(label_i+2)-*(label_j+2));
    int ij_diff_z = static_cast<int>(*(label_i+3)-*(label_j+3));
    int ik_diff_x = static_cast<int>(*(label_i+1)-*(label_k+1));
    int ik_diff_y = static_cast<int>(*(label_i+2)-*(label_k+2));
    int ik_diff_z = static_cast<int>(*(label_i+3)-*(label_k+3));

    int ij_x = ij_diff_x*ij_diff_x;
    int ij_y = ij_diff_y*ij_diff_y;
    int ij_z = ij_diff_z*ij_diff_z;
    int ik_x = ik_diff_x*ik_diff_x;
    int ik_y = ik_diff_y*ik_diff_y;
    int ik_z = ik_diff_z*ik_diff_z;

    int dist_ij = ij_x + ij_y + ij_z;
    int dist_ik = ik_x + ik_y + ik_z;
    if ((*label_i  == *label_j) && (*label_i  == *label_k))
      triplet_class_same = true;
    if ((dist_ij < dist_ik) && (triplet_class_same))
      triplet_pose_pass = true;
    if ((*label_i  == *label_j) && (*label_i  != *label_k))
      triplet_class_pass = true;
    if (*label_l == *label_m)
      pair_class_pass = true;
    if ((triplet_class_pass || triplet_pose_pass) && pair_class_pass) {
      datum.set_label(1);
      datum.SerializeToString(&value);
      snprintf(key, kMaxKeyLength, "%08d", itemid);
      db->Put(leveldb::WriteOptions(), std::string(key), value);
    } else {
      itemid--;
      datum.set_label(0);
    }
  }
  delete db;
  delete pixels;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("This script converts the MNIST dataset to the leveldb format used\n"
           "by caffe to train a siamese network.\n"
           "Usage:\n"
           "    convert_mnist_data input_image_file input_label_file "
           "output_db_file\n"
           "The MNIST dataset could be downloaded at\n"
           "    http://yann.lecun.com/exdb/mnist/\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(argv[1], argv[2], argv[3]);
  }
  return 0;
}
