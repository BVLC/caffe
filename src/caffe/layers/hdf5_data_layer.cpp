/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
HDF5DataLayer<Dtype>::~HDF5DataLayer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HDF5DataLayer<Dtype>::FillHDF5FileData() {
  int num_rows_filled = 0;
  while (true) {
    CHECK_LT(current_file_, hdf_filenames_.size());
    const char* filename = hdf_filenames_[current_file_].c_str();
    DLOG(INFO) << "Loading HDF5 file: " << filename;
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
      LOG(FATAL) << "Failed opening HDF5 file: " << filename;
    }
    int rows_read = -1;
    for (int i = 0; i < hdf_blobs_.size(); ++i) {
      const int current_rows_read = HDF5ReadRowsToBlob(
          file_id, this->layer_param_.top(i).c_str(),
          current_row_, num_rows_filled, hdf_blobs_[i].get());
      if (rows_read == -1) {
        CHECK_GE(current_rows_read, 0);
        rows_read = current_rows_read;
      }
      CHECK_EQ(rows_read, current_rows_read);
    }
    num_rows_filled += rows_read;
    CHECK_LE(num_rows_filled, hdf_blobs_[0]->num());
    herr_t status = H5Fclose(file_id);
    CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;
    DLOG(INFO) << "Successully loaded " << rows_read << " rows from: "
               << filename;
    // If we didn't fill up the blob, should move onto the next file.
    // If we did fill the blob, we may or may not be at the end.
    if (num_rows_filled < hdf_blobs_[0]->num()) {
      if (num_files_ > 1) {
        ++current_file_;
        if (current_file_ == num_files_) {
          current_file_ = 0;
          DLOG(INFO) << "Looping around to first file.";
        }
      }
      current_row_ = 0;
    } else {
      current_row_ += rows_read;
      break;
    }
  }
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Read the source to parse the filenames.
  const string& source = this->layer_param_.hdf5_data_param().source();
  LOG(INFO) << "Loading list of HDF5 filenames from: " << source;
  hdf_filenames_.clear();
  std::ifstream source_file(source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
      hdf_filenames_.push_back(line);
    }
  } else {
    LOG(FATAL) << "Failed to open source file: " << source;
  }
  CHECK_GT(hdf_filenames_.size(), 0)
      << "Source file must contain at least 1 filename: " << source;
  source_file.close();
  num_files_ = hdf_filenames_.size();
  current_file_ = 0;
  LOG(INFO) << "Number of HDF5 files: " << num_files_;
  CHECK_GE(num_files_, 1) << "Must have at least 1 HDF5 filename listed in "
    << source;

  // Reshape blobs.
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  const int top_size = this->layer_param_.top_size();
  hdf_blobs_.resize(top_size);
  hid_t file_id = H5Fopen(hdf_filenames_[0].c_str(), H5F_ACC_RDONLY,
                          H5P_DEFAULT);
  for (int i = 0; i < top_size; ++i) {
    hdf_blobs_[i].reset(new Blob<Dtype>(1, 1, 1, 1));
    HDF5PrepareBlob(file_id, this->layer_param_.top(i).c_str(), batch_size,
                    hdf_blobs_[i].get());
    hdf_blobs_[i]->mutable_cpu_data();
    top[i]->ReshapeLike(*hdf_blobs_[i]);
  }
  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << hdf_filenames_[0];

  Reset();

  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::Reset() {
  current_file_ = 0;
  current_row_ = 0;
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::InternalThreadEntry() {
  FillHDF5FileData();
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->JoinPrefetchThread();
  for (int i = 0; i < top.size(); ++i) {
    const int count = top[i]->count();
    caffe_copy(count, hdf_blobs_[i]->cpu_data(), top[i]->mutable_cpu_data());
  }
  this->CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(HDF5DataLayer, Forward);
#endif

INSTANTIATE_CLASS(HDF5DataLayer);
REGISTER_LAYER_CLASS(HDF5_DATA, HDF5DataLayer);
}  // namespace caffe
