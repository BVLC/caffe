// Copyright 2014 BVLC and contributors.

#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
using std::vector;

template <typename Dtype>
HDF5OutputLayer<Dtype>::HDF5OutputLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      file_name_(param.hdf5_output_param().file_name()) {
  /* create a HDF5 file */
  file_id_ = H5Fcreate(file_name_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                       H5P_DEFAULT);
  CHECK_GE(file_id_, 0) << "Failed to open HDF5 file" << file_name_;
}

template <typename Dtype>
HDF5OutputLayer<Dtype>::~HDF5OutputLayer<Dtype>() {
  herr_t status = H5Fclose(file_id_);
  CHECK_GE(status, 0) << "Failed to close HDF5 file " << file_name_;
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::SaveBlobs() {
  // TODO: no limit on the number of blobs
  LOG(INFO) << "Saving HDF5 file" << file_name_;
  CHECK_EQ(data_blob_.num(), label_blob_.num()) <<
      "data blob and label blob must have the same batch size";
  hdf5_save_nd_dataset(file_id_, HDF5_DATA_DATASET_NAME, data_blob_);
  hdf5_save_nd_dataset(file_id_, HDF5_DATA_LABEL_NAME, label_blob_);
  LOG(INFO) << "Successfully saved " << data_blob_.num() << " rows";
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // TODO: no limit on the number of blobs
  CHECK_EQ(bottom.size(), 2) << "HDF5OutputLayer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "HDF5OutputLayer takes no output blobs.";
}

template <typename Dtype>
Dtype HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_GE(bottom.size(), 2);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  data_blob_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                     bottom[0]->height(), bottom[0]->width());
  label_blob_.Reshape(bottom[1]->num(), bottom[1]->channels(),
                     bottom[1]->height(), bottom[1]->width());
  const int data_datum_dim = bottom[0]->count() / bottom[0]->num();
  const int label_datum_dim = bottom[1]->count() / bottom[1]->num();

  for (int i = 0; i < bottom[0]->num(); ++i) {
    memcpy(&data_blob_.mutable_cpu_data()[i * data_datum_dim],
           &bottom[0]->cpu_data()[i * data_datum_dim],
           sizeof(Dtype) * data_datum_dim);
    memcpy(&label_blob_.mutable_cpu_data()[i * label_datum_dim],
           &bottom[1]->cpu_data()[i * label_datum_dim],
           sizeof(Dtype) * label_datum_dim);
  }
  SaveBlobs();
  return Dtype(0.);
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return;
}

INSTANTIATE_CLASS(HDF5OutputLayer);

}  // namespace caffe
