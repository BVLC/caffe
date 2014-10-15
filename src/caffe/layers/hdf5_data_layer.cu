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
void HDF5DataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    const int count = top[i]->count();
    caffe_copy(count, hdf_blobs_[i]->gpu_data(), top[i]->mutable_gpu_data());
  }
  this->InternalThreadEntry();
}

INSTANTIATE_LAYER_GPU_FORWARD(HDF5DataLayer);

}  // namespace caffe
