/*
Contributors:
- Sergey Karayev, 2014.
- Tobias Domhan, 2014.

TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
*/
#include <stdint.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
HDF5DataLayer<Dtype>::~HDF5DataLayer<Dtype>() { }

template <typename Dtype>
void HDF5DataLayer<Dtype>::load_hdf5_file(const char* filename) {
  LOG(INFO) << "Loading HDF5 file" << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(ERROR) << "Failed opening HDF5 file" << filename;
    return;
  }
  const int MIN_DATA_DIM = 2;
  const int MAX_DATA_DIM = 4;
  const int MIN_LABEL_DIM = 1;
  const int MAX_LABEL_DIM = 2;
  hdf5_load_nd_dataset(file_id, "data",  MIN_DATA_DIM, MAX_DATA_DIM,
                      &data_,  data_dims_);
  hdf5_load_nd_dataset(file_id, "label", MIN_LABEL_DIM, MAX_LABEL_DIM,
                      &label_, label_dims_);

  // Add missing dimensions.
  const int MAX_BLOB_DIM = 4;
  while(data_dims_.size() < MAX_BLOB_DIM) {
    data_dims_.push_back(1);
  }
  while(label_dims_.size() < MAX_BLOB_DIM) {
    label_dims_.push_back(1);
  }

  herr_t status = H5Fclose(file_id);
  CHECK_EQ(data_dims_[0], label_dims_[0]);
  LOG(INFO) << "Successully loaded " << data_dims_[0] << " rows";
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "HDF5DataLayer takes no input blobs.";
  CHECK_EQ(top->size(), 2) << "HDF5DataLayer takes two blobs as output.";

  // Read the source to parse the filenames.
  LOG(INFO) << "Loading filename from " << this->layer_param_.source();
  hdf_filenames_.clear();
  std::ifstream myfile(this->layer_param_.source().c_str());
  if (myfile.is_open()) {
    string line = "";
    while (myfile >> line) {
      hdf_filenames_.push_back(line);
    }
  }
  myfile.close();
  num_files_ = hdf_filenames_.size();
  current_file_ = 0;
  LOG(INFO) << "Number of files: " << num_files_;

  // Load the first HDF5 file and initialize the line counter.
  load_hdf5_file(hdf_filenames_[current_file_].c_str());
  current_row_ = 0;

  // Reshape blobs.
  (*top)[0]->Reshape(this->layer_param_.batchsize(),
                     data_dims_[1], data_dims_[2], data_dims_[3]);
  (*top)[1]->Reshape(this->layer_param_.batchsize(),
                     label_dims_[1], label_dims_[2], label_dims_[3]);
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int batchsize = this->layer_param_.batchsize();
  const int data_count = (*top)[0]->count() / (*top)[0]->num();
  const int label_data_count = (*top)[1]->count() / (*top)[1]->num();

  for (int i = 0; i < batchsize; ++i, ++current_row_) {
    if (current_row_ == data_dims_[0]) {
      if (num_files_ > 1) {
        current_file_ += 1;

        if (current_file_ == num_files_) {
          current_file_ = 0;
          LOG(INFO) << "looping around to first file";
        }

        load_hdf5_file(hdf_filenames_[current_file_].c_str());
      }
      current_row_ = 0;
    }

    memcpy(&(*top)[0]->mutable_cpu_data()[i * data_count],
            &(data_.get()[current_row_ * data_count]),
            sizeof(Dtype) * data_count);

    memcpy(&(*top)[1]->mutable_cpu_data()[i * label_data_count],
            &(label_.get()[current_row_ * label_data_count]),
            sizeof(Dtype) * label_data_count);
  }
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype HDF5DataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(HDF5DataLayer);

}  // namespace caffe
