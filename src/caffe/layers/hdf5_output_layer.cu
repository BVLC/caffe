#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/hdf5_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void HDF5OutputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom.size(), 1);

  for (int i = 0; i < bottom.size(); ++i) {
    CHECK_EQ(bottom[0]->num(), bottom[i]->num());
    data_blobs_[i]->Reshape(bottom[i]->num(), bottom[i]->channels(),
        bottom[i]->height(), bottom[i]->width());
    caffe_copy(bottom[i]->count(), bottom[i]->gpu_data(),
        data_blobs_[i]->mutable_cpu_data());
  }
  SaveBlobs();
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

INSTANTIATE_LAYER_GPU_FUNCS(HDF5OutputLayer);

}  // namespace caffe
