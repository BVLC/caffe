#include <vector>

#include "caffe/layers/filter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void FilterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  int_tp new_tops_num = indices_to_forward_.size();
  // forward all filtered items for all bottoms but the Selector (bottom[last])
  for (int_tp t = 0; t < top.size(); ++t) {
    const Dtype* bottom_data = bottom[t]->gpu_data();
    Dtype* top_data = top[t]->mutable_gpu_data();
    int_tp dim = bottom[t]->count() / bottom[t]->shape(0);
    for (int_tp n = 0; n < new_tops_num; ++n) {
      int_tp data_offset_top = n * dim;
      int_tp data_offset_bottom = indices_to_forward_[n] * dim;

      if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
        caffe_copy(dim, bottom_data + data_offset_bottom,
                   top_data + data_offset_top);
#endif  // USE_CUDA
      } else {
#ifdef USE_GREENTEA
        viennacl::ocl::context &ctx = viennacl::ocl::get_context(
            this->device_->id());

        greentea_copy<Dtype>(dim, (cl_mem) bottom_data, data_offset_bottom,
                             (cl_mem) top_data, data_offset_top, &ctx);
#endif  // USE_GREENTEA
      }
    }
  }
}

template<typename Dtype>
void FilterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[bottom.size() - 1]) {
    LOG(FATAL)<< this->type()
    << "Layer cannot backpropagate to filter index inputs";
  }
  for (int_tp i = 0; i < top.size(); ++i) {
    // bottom[last] is the selector and never needs backpropagation
    // so we can iterate over top vector because top.size() == bottom.size() -1
    if (propagate_down[i]) {
      const int_tp dim = top[i]->count() / top[i]->shape(0);
      int_tp next_to_backward_offset = 0;
      int_tp batch_offset = 0;
      int_tp data_offset_bottom = 0;
      int_tp data_offset_top = 0;

      if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
        for (int_tp n = 0; n < bottom[i]->shape(0); ++n) {
          if (next_to_backward_offset >= indices_to_forward_.size()) {
            // we already visited all items that were been forwarded, so
            // just set to zero remaining ones
            data_offset_bottom = n * dim;
            caffe_gpu_set(dim, Dtype(0),
                bottom[i]->mutable_gpu_diff() + data_offset_bottom);
          } else {
            batch_offset = indices_to_forward_[next_to_backward_offset];
            data_offset_bottom = n * dim;
            if (n != batch_offset) {  // this data was not been forwarded
              caffe_gpu_set(dim, Dtype(0),
                  bottom[i]->mutable_gpu_diff() + data_offset_bottom);
            } else {  // this data was been forwarded
              data_offset_top = next_to_backward_offset * dim;
              ++next_to_backward_offset;  // point to next forwarded item index
              caffe_copy(dim, top[i]->mutable_gpu_diff() + data_offset_top,
                  bottom[i]->mutable_gpu_diff() + data_offset_bottom);
            }
          }
        }
#endif  // USE_CUDA
      } else {
#ifdef USE_GREENTEA
        viennacl::ocl::context &ctx = viennacl::ocl::get_context(
            this->device_->id());

        for (int_tp n = 0; n < bottom[i]->shape(0); ++n) {
          if (next_to_backward_offset >= indices_to_forward_.size()) {
            // we already visited all items that were been forwarded, so
            // just set to zero remaining ones
            data_offset_bottom = n * dim;
            greentea_gpu_set(this->device_->id(), dim, Dtype(0),
                (cl_mem)(bottom[i]->mutable_gpu_diff()), data_offset_bottom);
          } else {
            batch_offset = indices_to_forward_[next_to_backward_offset];
            data_offset_bottom = n * dim;
            if (n != batch_offset) {  // this data was not been forwarded
              greentea_gpu_set(this->device_->id(), dim, Dtype(0),
                  (cl_mem)(bottom[i]->mutable_gpu_diff()), data_offset_bottom);
            } else {  // this data was been forwarded
              data_offset_top = next_to_backward_offset * dim;
              ++next_to_backward_offset;  // point to next forwarded item index
              greentea_copy<Dtype>(dim, (cl_mem)(top[i]->mutable_gpu_diff()),
                  data_offset_top,
                  (cl_mem)(bottom[i]->mutable_gpu_diff()),
                  data_offset_bottom, &ctx);
            }
          }
        }
#endif  // USE_GREENTEA
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FilterLayer);

}  // namespace caffe
