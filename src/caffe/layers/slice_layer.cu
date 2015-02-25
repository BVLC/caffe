#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SliceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->mutable_gpu_data();
  if (slice_dim_ == 0) {
    int offset_num = 0;
    for (int i = 0; i < top->size(); ++i) {
      Blob<Dtype>* blob = (*top)[i];
      Dtype* top_data = blob->mutable_gpu_data();
      caffe_copy(blob->count(), bottom_data + bottom[0]->offset(offset_num),
                 top_data);
      offset_num += blob->num();
    }
  } else if (slice_dim_ == 1) {
    int offset_channel = 0;
    for (int i = 0; i < top->size(); ++i) {
      Blob<Dtype>* blob = (*top)[i];
      Dtype* top_data = blob->mutable_gpu_data();
      const int num_elem = blob->channels() * blob->height() * blob->width();
      for (int n = 0; n < num_; ++n) {
        caffe_copy(num_elem, bottom_data + bottom[0]->offset(n, offset_channel),
                   top_data + blob->offset(n));
      }
      offset_channel += blob->channels();
    }
  }  // slice_dim_ is guaranteed to be 0 or 1 by SetUp.
}

template <typename Dtype>
void SliceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) { return; }
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  if (slice_dim_ == 0) {
    int offset_num = 0;
    for (int i = 0; i < top.size(); ++i) {
      Blob<Dtype>* blob = top[i];
      const Dtype* top_diff = blob->gpu_diff();
      caffe_copy(blob->count(), top_diff,
                 bottom_diff + (*bottom)[0]->offset(offset_num));
      offset_num += blob->num();
    }
  } else if (slice_dim_ == 1) {
    int offset_channel = 0;
    for (int i = 0; i < top.size(); ++i) {
      Blob<Dtype>* blob = top[i];
      const Dtype* top_diff = blob->gpu_diff();
      const int num_elem = blob->channels() * blob->height() * blob->width();
      for (int n = 0; n < num_; ++n) {
        caffe_copy(num_elem, top_diff + blob->offset(n),
                   bottom_diff +  (*bottom)[0]->offset(n, offset_channel));
      }
      offset_channel += blob->channels();
    }
  }  // slice_dim_ is guaranteed to be 0 or 1 by SetUp.
}

INSTANTIATE_CLASS(SliceLayer);

}  // namespace caffe
