// Usage:
// convert_3d_data input_image_file input_label_file output_db_file
// Codes are disigned for binary files including data and label. You can modify
// the condition if information for arranging training data is not the same with
// category and pose of object.
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#ifdef USE_LEVELDB
#include "leveldb/db.h"
#include "math.h"
#include "stdint.h"

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void read_image(std::ifstream* image_file, std::ifstream* label_file,
        uint32_t index, uint32_t rows, uint32_t cols,
        char* pixels, char* label_temp, signed char* label, int rgb_use) {
  if (rgb_use == 0) {
    image_file->seekg(index * rows * cols + 16);
    image_file->read(pixels, rows * cols);
    label_file->seekg(index * 4 + 8);  // 4 = 1 catory label+3 coordinate label
    label_file->read(label_temp, 4);
    for (int i = 0; i < 4; i++)
      *(label+i) = (signed char)*(label_temp+i);
  } else {
    image_file->seekg(3 * index * rows * cols + 16);
    image_file->read(pixels, 3 * rows * cols);
    label_file->seekg(index * 4 + 8);  // 4 = 1 catory label+3 coordinate label
    label_file->read(label_temp, 4);
    for (int i = 0; i < 4; i++)
      *(label+i) = (signed char)*(label_temp+i);
  }
}

void convert_dataset(const char* image_filename, const char* label_filename,
        const char* db_filename,
                     const char* class_number, const char* rgb_use) {
  int rgb_use1 = atoi(rgb_use);
  int class_num = atoi(class_number);
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
  int db_size;
  if (rgb_use1 == 0)
    db_size = rows * cols;
  else
    db_size = 3 * rows * cols;
  char* pixels1 = new char[db_size];
  char* pixels2 = new char[db_size];
  char* pixels3 = new char[db_size];
  char* pixels4 = new char[db_size];
  char* pixels5 = new char[db_size];
  const int kMaxKeyLength = 10;
  char key[kMaxKeyLength];
  std::string value;
  caffe::Datum datum;
  if (rgb_use1 == 0)
    datum.set_channels(1);
  else
    datum.set_channels(3);
  datum.set_height(rows);
  datum.set_width(cols);
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  int counter = 0;
  // This codes selecting 1 positive sample and 3 negative samples for a triplet
  // set. We randomly select data and decide whether concatenating data set to
  // DB file according to labels.
  for (unsigned int times = 0; times < 10; ++times) {
    // iteration in the samples of all class
    for (unsigned int itemid = 0; itemid < num_items/class_num; ++itemid) {
      // iteration in the samples in one class
      for (unsigned int class_ind = 0; class_ind < class_num; ++class_ind) {
      // use reference sample one by one at each iteration
      int i = itemid % num_items + class_ind*num_items/class_num;
      int j = caffe::caffe_rng_rand() % num_items;  // pick triplet groups
      int k = caffe::caffe_rng_rand() % num_items;
      int l = caffe::caffe_rng_rand() % num_items;  // pick pair wise groups
      int m = caffe::caffe_rng_rand() % num_items;
      read_image(&image_file, &label_file, i, rows, cols,  // read triplet
        pixels1, label_temp, label_i, rgb_use1);
      read_image(&image_file, &label_file, j, rows, cols,
        pixels2, label_temp, label_j, rgb_use1);
      read_image(&image_file, &label_file, k, rows, cols,
        pixels3, label_temp, label_k, rgb_use1);
      read_image(&image_file, &label_file, l, rows, cols,  // read pair wise
        pixels4, label_temp, label_l, rgb_use1);
      read_image(&image_file, &label_file, m, rows, cols,
        pixels5, label_temp, label_m, rgb_use1);

      bool pair_pass = false;
      bool triplet1_pass = false;
      bool triplet2_pass = false;
      bool triplet3_class_same = false;
      bool triplet3_pass = false;

      int ij_diff_x = static_cast<int>(*(label_i+1)-*(label_j+1));
      int ij_diff_y = static_cast<int>(*(label_i+2)-*(label_j+2));
      int ij_diff_z = static_cast<int>(*(label_i+3)-*(label_j+3));
      int im_diff_x = static_cast<int>(*(label_i+1)-*(label_m+1));
      int im_diff_y = static_cast<int>(*(label_i+2)-*(label_m+2));
      int im_diff_z = static_cast<int>(*(label_i+3)-*(label_m+3));

      int ij_x = ij_diff_x*ij_diff_x;
      int ij_y = ij_diff_y*ij_diff_y;
      int ij_z = ij_diff_z*ij_diff_z;
      int im_x = im_diff_x*im_diff_x;
      int im_y = im_diff_y*im_diff_y;
      int im_z = im_diff_z*im_diff_z;

      float dist_ij = std::sqrt(ij_x + ij_y + ij_z);
      float dist_im = std::sqrt(im_x + im_y + im_z);
      // Arrange training data according to conditionals including category
      // and pose of synthetic data, dist_* could be ignored if you
      // only concentrate on category.
      if (*label_i == *label_j && dist_ij < 100/3 && dist_ij != 0)
        pair_pass = true;
      if (pair_pass && (*label_i  != *label_k))
        triplet1_pass = true;
      if (pair_pass && (*label_i  != *label_l))
        triplet2_pass = true;
      if (pair_pass && (*label_i  == *label_m))
        triplet3_class_same = true;
      if (triplet3_class_same && dist_im > 100/3)
        triplet3_pass = true;
      if (pair_pass && triplet1_pass && triplet2_pass && triplet3_pass) {
        datum.set_data(pixels1, db_size);  // set data
        datum.set_label(static_cast<int>(*label_i));
        datum.SerializeToString(&value);
        snprintf(key, kMaxKeyLength, "%08d", counter);
        db->Put(leveldb::WriteOptions(), std::string(key), value);
        counter++;
        datum.set_data(pixels2, db_size);  // set data
        datum.set_label(static_cast<int>(*label_j));
        datum.SerializeToString(&value);
        snprintf(key, kMaxKeyLength, "%08d", counter);
        db->Put(leveldb::WriteOptions(), std::string(key), value);
        counter++;
        datum.set_data(pixels3, db_size);  // set data
        datum.set_label(static_cast<int>(*label_k));
        datum.SerializeToString(&value);
        snprintf(key, kMaxKeyLength, "%08d", counter);
        db->Put(leveldb::WriteOptions(), std::string(key), value);
        counter++;
        datum.set_data(pixels4, db_size);  // set data
        datum.set_label(static_cast<int>(*label_l));
        datum.SerializeToString(&value);
        snprintf(key, kMaxKeyLength, "%08d", counter);
        db->Put(leveldb::WriteOptions(), std::string(key), value);
        counter++;
        datum.set_data(pixels5, db_size);  // set data
        datum.set_label(static_cast<int>(*label_m));
        datum.SerializeToString(&value);
        snprintf(key, kMaxKeyLength, "%08d", counter);
        db->Put(leveldb::WriteOptions(), std::string(key), value);
        counter++;
      } else {
        class_ind--;
      }
      }  // iteration in the samples of all class
    }  // iteration in the samples in one class
  }  // iteration in times
  delete db;
  delete pixels1;
  delete pixels2;
  delete pixels3;
  delete pixels4;
  delete pixels5;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    printf("This script converts the dataset to the leveldb format used\n"
           "by caffe to train a triplet network.\n"
           "Usage:\n"
           "    convert_3d_data input_image_file input_label_file "
           "output_db_file class_number rgb_use \n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(argv[1], argv[2], argv[3], argv[4], argv[5]);
  }
  return 0;
}
#else
int main(int argc, char** argv) {
    LOG(FATAL) << "This example requires LevelDB; compile with USE_LEVELDB.";
}
#endif  // USE_LEVELDB
