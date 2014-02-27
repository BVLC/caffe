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

template <typename Dtype>
Dtype HDF5DataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(HDF5DataLayer);

}  // namespace caffe
