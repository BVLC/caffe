// Copyright Sergey Karayev 2014
/*
TODO:
- only load parts of the file, in accordance with a prototxt param "max_mem"
*/

#include <stdint.h>
#include <string>
#include <vector>

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
void HDF5DataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "HDF5DataLayer takes no input blobs.";
  CHECK_EQ(top->size(), 2) << "HDF5DataLayer takes two blobs as output.";

  // Load the HDF5 file and initialize the counter.
  const char* hdf_filename = this->layer_param_.source().c_str();
  LOG(INFO) << "Loading HDF5 file" << hdf_filename;
  hid_t file_id = H5Fopen(hdf_filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(ERROR) << "Failed opening HDF5 file" << hdf_filename;
    return;
  }
  const int MAX_DATA_DIM = 4;
  const int MAX_LABEL_DIM = 2;
  const int MIN_DIM = 2;
  hd5_load_nd_dataset(file_id, "data",  MIN_DIM, MAX_DATA_DIM,
                      &data_,  data_dims_);
  hd5_load_nd_dataset(file_id, "label", MIN_DIM, MAX_LABEL_DIM,
                      &label_, label_dims_);

  while(data_dims_.size() < MAX_DATA_DIM) {
    data_dims_.push_back(1);
  }

  //add missing dimensions:
  label_dims_.push_back(1);
  label_dims_.push_back(1);

  herr_t status = H5Fclose(file_id);
  CHECK_EQ(data_dims_[0], label_dims_[0]);
  LOG(INFO) << "Successully loaded " << data_dims_[0] << " rows";
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

  //TODO: consolidate into a single memcpy call

  for (int i = 0; i < batchsize; ++i, ++current_row_) {
    if (current_row_ == data_dims_[0]) {
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
