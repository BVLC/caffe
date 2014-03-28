// Copyright 2014 BVLC and contributors.
/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <stdint.h>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
HDF5DataLayer<Dtype>::~HDF5DataLayer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HDF5DataLayer<Dtype>::LoadHDF5FileData(const char* filename) {
  LOG(INFO) << "Loading HDF5 file" << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(ERROR) << "Failed opening HDF5 file" << filename;
    return;
  }

  const int MIN_DATA_DIM = 2;
  const int MAX_DATA_DIM = 4;
  hdf5_load_nd_dataset(
    file_id, "data",  MIN_DATA_DIM, MAX_DATA_DIM, &data_blob_);

  const int MIN_LABEL_DIM = 1;
  const int MAX_LABEL_DIM = 2;
  hdf5_load_nd_dataset(
    file_id, "label", MIN_LABEL_DIM, MAX_LABEL_DIM, &label_blob_);

  herr_t status = H5Fclose(file_id);
  CHECK_EQ(data_blob_.num(), label_blob_.num());
  LOG(INFO) << "Successully loaded " << data_blob_.num() << " rows";
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "HDF5DataLayer takes no input blobs.";
  CHECK_EQ(top->size(), 2) << "HDF5DataLayer takes two blobs as output.";

  // Read the source to parse the filenames.
  const string& source = this->layer_param_.hdf5_data_param().source();
  LOG(INFO) << "Loading filename from " << source;
  hdf_filenames_.clear();
  std::ifstream source_file(source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
      hdf_filenames_.push_back(line);
    }
  }
  source_file.close();
  num_files_ = hdf_filenames_.size();
  current_file_ = 0;
  LOG(INFO) << "Number of files: " << num_files_;

  // Load the first HDF5 file and initialize the line counter.
  LoadHDF5FileData(hdf_filenames_[current_file_].c_str());
  current_row_ = 0;

  // Reshape blobs.
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  (*top)[0]->Reshape(batch_size, data_blob_.channels(),
                     data_blob_.width(), data_blob_.height());
  (*top)[1]->Reshape(batch_size, label_blob_.channels(),
                     label_blob_.width(), label_blob_.height());
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
}

template <typename Dtype>
Dtype HDF5DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  const int data_count = (*top)[0]->count() / (*top)[0]->num();
  const int label_data_count = (*top)[1]->count() / (*top)[1]->num();

  for (int i = 0; i < batch_size; ++i, ++current_row_) {
    if (current_row_ == data_blob_.num()) {
      if (num_files_ > 1) {
        current_file_ += 1;
        if (current_file_ == num_files_) {
          current_file_ = 0;
          LOG(INFO) << "looping around to first file";
        }
        LoadHDF5FileData(hdf_filenames_[current_file_].c_str());
      }
      current_row_ = 0;
    }
    memcpy(&(*top)[0]->mutable_cpu_data()[i * data_count],
           &data_blob_.cpu_data()[current_row_ * data_count],
           sizeof(Dtype) * data_count);
    memcpy(&(*top)[1]->mutable_cpu_data()[i * label_data_count],
            &label_blob_.cpu_data()[current_row_ * label_data_count],
            sizeof(Dtype) * label_data_count);
  }
  return Dtype(0.);
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
void HDF5DataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { }

INSTANTIATE_CLASS(HDF5DataLayer);

}  // namespace caffe
