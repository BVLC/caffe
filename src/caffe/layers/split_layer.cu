#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SplitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < top->size(); ++i) {
    (*top)[i]->ShareData(*bottom[0]);
  }
}

template <typename Dtype>
void SplitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) { return; }
  if (top.size() == 1) {
    caffe_copy(count_, top[0]->gpu_diff(), (*bottom)[0]->mutable_gpu_diff());
    return;
  }
  caffe_gpu_add(count_, top[0]->gpu_diff(), top[1]->gpu_diff(),
                (*bottom)[0]->mutable_gpu_diff());
  // Add remaining top blob diffs.
  for (int i = 2; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    caffe_gpu_axpy(count_, Dtype(1.), top_diff, bottom_diff);
  }
}


INSTANTIATE_CLASS(SplitLayer);

}  // namespace caffe
