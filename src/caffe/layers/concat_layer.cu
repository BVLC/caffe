#include <vector>

#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Forward_const_gpu(bottom,top);
}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_const_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const {
  if (bottom.size() == 1) { return; }
  auto concat_axis= get_concat_axis(bottom);
  int num_concats = bottom[0]->count(0, concat_axis);
  Dtype* top_data = top[0]->mutable_gpu_data();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis);
  int concat_input_size = bottom[0]->count(concat_axis + 1);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis);

    //我们直接使用内存复制更快

    for (int n = 0; n < num_concats; ++n) {
      caffe_copy(bottom_concat_axis * concat_input_size,
          bottom_data + n * bottom_concat_axis * concat_input_size,
          top_data + (n * top_concat_axis + offset_concat_axis)
              * concat_input_size);
    }
    offset_concat_axis += bottom_concat_axis;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS_CONST(ConcatLayer);

}  // namespace caffe
