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
Dtype HDF5OutputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
    CUDA_CHECK(cudaMemcpy(&data_blob_.mutable_cpu_data()[i * data_datum_dim],
           &bottom[0]->gpu_data()[i * data_datum_dim],
           sizeof(Dtype) * data_datum_dim, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&label_blob_.mutable_cpu_data()[i * label_datum_dim],
           &bottom[1]->gpu_data()[i * label_datum_dim],
           sizeof(Dtype) * label_datum_dim, cudaMemcpyDeviceToHost));
  }
  SaveBlobs();
  return Dtype(0.);
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return;
}

INSTANTIATE_CLASS(HDF5OutputLayer);

}  // namespace caffe
