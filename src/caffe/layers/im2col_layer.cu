#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void Im2colLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  im2col_gpu(bottom[0]->gpu_data(),
     bottom[0]->num(), channels_, height_, width_,
     kernel_h_, kernel_w_, pad_h_, pad_w_,
     stride_h_, stride_w_, hole_h_, hole_w_,
     top[0]->mutable_gpu_data());
}

template <typename Dtype>
void Im2colLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  col2im_gpu(top[0]->gpu_diff(),
      top[0]->num(), channels_, height_, width_,
      kernel_h_, kernel_w_, pad_h_, pad_w_,
      stride_h_, stride_w_, hole_h_, hole_w_,
      bottom[0]->mutable_gpu_diff());
}


INSTANTIATE_LAYER_GPU_FUNCS(Im2colLayer);

}  // namespace caffe
