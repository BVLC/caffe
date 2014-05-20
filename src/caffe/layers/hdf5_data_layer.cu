// Copyright 2014 BVLC and contributors.
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
Dtype HDF5DataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
    CUDA_CHECK(cudaMemcpy(
            &(*top)[0]->mutable_gpu_data()[i * data_count],
            &data_blob_.cpu_data()[current_row_ * data_count],
            sizeof(Dtype) * data_count,
            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
            &(*top)[1]->mutable_gpu_data()[i * label_data_count],
            &label_blob_.cpu_data()[current_row_ * label_data_count],
            sizeof(Dtype) * label_data_count,
            cudaMemcpyHostToDevice));
  }
  return Dtype(0.);
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
}

INSTANTIATE_CLASS(HDF5DataLayer);

}  // namespace caffe
