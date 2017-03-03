/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/hdf5_output_layer.hpp"
#include "caffe/util/hdf5.hpp"

namespace caffe {

template <typename Dtype>
void HDF5OutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  file_name_ = this->layer_param_.hdf5_output_param().file_name();
  file_id_ = H5Fcreate(file_name_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                       H5P_DEFAULT);
  CHECK_GE(file_id_, 0) << "Failed to open HDF5 file" << file_name_;
  file_opened_ = true;
}

template <typename Dtype>
HDF5OutputLayer<Dtype>::~HDF5OutputLayer<Dtype>() {
  if (file_opened_) {
    herr_t status = H5Fclose(file_id_);
    CHECK_GE(status, 0) << "Failed to close HDF5 file " << file_name_;
  }
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::SaveBlobs() {
  // TODO: no limit on the number of blobs
  LOG(INFO) << "Saving HDF5 file " << file_name_;
  CHECK_EQ(data_blob_.num(), label_blob_.num()) <<
      "data blob and label blob must have the same batch size";
  hdf5_save_nd_dataset(file_id_, HDF5_DATA_DATASET_NAME, data_blob_);
  hdf5_save_nd_dataset(file_id_, HDF5_DATA_LABEL_NAME, label_blob_);
  LOG(INFO) << "Successfully saved " << data_blob_.num() << " rows";
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom.size(), 2);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  data_blob_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                     bottom[0]->height(), bottom[0]->width());
  label_blob_.Reshape(bottom[1]->num(), bottom[1]->channels(),
                     bottom[1]->height(), bottom[1]->width());
  const int data_datum_dim = bottom[0]->count() / bottom[0]->num();
  const int label_datum_dim = bottom[1]->count() / bottom[1]->num();

  for (int i = 0; i < bottom[0]->num(); ++i) {
    caffe_copy(data_datum_dim, &bottom[0]->cpu_data()[i * data_datum_dim],
        &data_blob_.mutable_cpu_data()[i * data_datum_dim]);
    caffe_copy(label_datum_dim, &bottom[1]->cpu_data()[i * label_datum_dim],
        &label_blob_.mutable_cpu_data()[i * label_datum_dim]);
  }
  SaveBlobs();
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

#ifdef CPU_ONLY
STUB_GPU(HDF5OutputLayer);
#endif

INSTANTIATE_CLASS(HDF5OutputLayer);
REGISTER_LAYER_CLASS(HDF5Output);

}  // namespace caffe
