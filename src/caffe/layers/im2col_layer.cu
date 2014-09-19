#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void Im2colLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
        width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
        stride_h_, stride_w_, top_data + (*top)[0]->offset(n));
  }
}

template <typename Dtype>
void Im2colLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  for (int n = 0; n < top[0]->num(); ++n) {
    col2im_gpu(top_diff + top[0]->offset(n), channels_, height_, width_,
        kernel_h_, kernel_w_, pad_h_, pad_w_,
        stride_h_, stride_w_, bottom_diff + (*bottom)[0]->offset(n));
  }
}


INSTANTIATE_CLASS(Im2colLayer);

}  // namespace caffe
