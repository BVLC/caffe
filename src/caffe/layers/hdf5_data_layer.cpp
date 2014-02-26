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
  load_2d_dataset(file_id, "data", &data, data_dims);
  load_2d_dataset(file_id, "label", &label, label_dims);
  herr_t status = H5Fclose(file_id);
  assert(data_dims[0] == label_dims[0]);
  current_row = 0;

  // Reshape blobs.
  (*top)[0]->Reshape(this->layer_param_.batchsize(), data_dims[1], 1, 1);
  (*top)[1]->Reshape(this->layer_param_.batchsize(), label_dims[1], 1, 1);
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int batchsize = this->layer_param_.batchsize();
  for (int i = 0; i < batchsize; ++i, ++current_row) {
    if (current_row == data_dims[0]) {
      current_row = 0;
    }

    memcpy(&(*top)[0]->mutable_cpu_data()[i * data_dims[1]],
            &(data.get()[current_row * data_dims[1]]),
            sizeof(Dtype) * data_dims[1]);

    memcpy(&(*top)[1]->mutable_cpu_data()[i * label_dims[1]],
            &(label.get()[current_row * label_dims[1]]),
            sizeof(Dtype) * label_dims[1]);
  }
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int batchsize = this->layer_param_.batchsize();
  for (int i = 0; i < batchsize; ++i, ++current_row) {
    if (current_row == data_dims[0]) {
      current_row = 0;
    }

    CUDA_CHECK(cudaMemcpy(
            &(*top)[0]->mutable_gpu_data()[i * data_dims[1]],
            &(data.get()[current_row * data_dims[1]]),
            sizeof(Dtype) * data_dims[1],
            cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
            &(*top)[1]->mutable_gpu_data()[i * label_dims[1]],
            &(label.get()[current_row * label_dims[1]]),
            sizeof(Dtype) * label_dims[1],
            cudaMemcpyHostToDevice));
  }
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype HDF5DataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

template <typename Dtype>
Dtype HDF5DataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(HDF5DataLayer);

}  // namespace caffe
