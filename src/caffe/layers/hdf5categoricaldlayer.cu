#include <stdint.h>
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

int compute_blocksize(int target, int limit){
  limit=min(target,limit);
  int blocksize=1;
  while (blocksize < limit) blocksize <<= 1;
  return blocksize;
}


template <typename Dtype>
void HDF5CategoricalDLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // do the one hot encoding work on the cpu
  Forward_cpu(bottom,top);

  const int batch_size = 
    this->layer_param_.hdf5_categorical_data_param().batch_size();
  const int top_data_count = top[0]->count() / top[0]->num();
  const int label_data_count = top[1]->count() / top[1]->num();

   
    caffe_copy(top_data_count*batch_size,
	       top[0]->cpu_data(),
	       top[0]->mutable_gpu_data());
    caffe_copy(label_data_count*batch_size,
	       top[1]->cpu_data(),
               top[1]->mutable_gpu_data());
}


  INSTANTIATE_LAYER_GPU_FUNCS(HDF5CategoricalDLayer);

}  // namespace caffe
