#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void NumpyDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

INSTANTIATE_LAYER_GPU_FUNCS(NumpyDataLayer);

}  // namespace caffe
