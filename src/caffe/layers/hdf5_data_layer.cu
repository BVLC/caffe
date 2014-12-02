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

namespace caffe {

template <typename Dtype>
void HDF5DataLayer<Dtype>::PermutateData(const hsize_t max_val){
  LOG(INFO) << "shuffle data";
  std::random_shuffle(permutation_.begin(), permutation_.end());
}


template <typename Dtype>
void HDF5DataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  for (int i = 0; i < batch_size; ++i, ++current_row_) {
    if (current_row_ == hdf_blobs_[0]->num()) {
      if (num_files_ > 1) {
        current_file_ += 1;
        if (current_file_ == num_files_) {
          current_file_ = 0;
          DLOG(INFO) << "Looping around to first file.";
        }
        LoadHDF5FileData(hdf_filenames_[current_file_].c_str());
        if(this->layer_param_.hdf5_data_param().shuffle())
          ShuffleData(hdf_blobs_[0]->num());
      }
      current_row_ = 0;
      if(this->layer_param_.hdf5_data_param().shuffle())
        ShuffleData(hdf_blobs_[0]->num());
    }
    for (int j = 0; j < this->layer_param_.top_size(); ++j) {
      int data_dim = top[j]->count() / top[j]->num();
      caffe_copy(data_dim,
          &hdf_blobs_[j]->cpu_data()[permutation_[current_row_] * data_dim],
          &top[j]->mutable_gpu_data()[i * data_dim]);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(HDF5DataLayer);

}  // namespace caffe
