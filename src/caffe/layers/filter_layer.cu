#include <vector>

#include "caffe/layers/filter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void FilterLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                     const vector<Blob<MItype>*>& bottom,
                                     const vector<Blob<MOtype>*>& top) {
  int_tp new_tops_num = indices_to_forward_.size();
  // forward all filtered items for all bottoms but the Selector (bottom[last])
  for (int_tp t = 0; t < top.size(); ++t) {
    vptr<const Dtype> bottom_data = bottom[t]->gpu_data();
    vptr<Dtype> top_data = top[t]->mutable_gpu_data();
    int_tp dim = bottom[t]->count() / bottom[t]->shape(0);
    for (int_tp n = 0; n < new_tops_num; ++n) {
      int_tp data_offset_top = n * dim;
      int_tp data_offset_bottom = indices_to_forward_[n] * dim;

      this->device_->template copy<Dtype>(dim, bottom_data + data_offset_bottom,
                                          top_data + data_offset_top);
    }
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void FilterLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                      const vector<Blob<MOtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<MItype>*>& bottom) {
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

      for (int_tp n = 0; n < bottom[i]->shape(0); ++n) {
        if (next_to_backward_offset >= indices_to_forward_.size()) {
          // we already visited all items that were been forwarded, so
          // just set to zero remaining ones
          data_offset_bottom = n * dim;
          this->device_->set(dim, Dtype(0),
              bottom[i]->mutable_gpu_diff() + data_offset_bottom);
        } else {
          batch_offset = indices_to_forward_[next_to_backward_offset];
          data_offset_bottom = n * dim;
          if (n != batch_offset) {  // this data was not been forwarded
            this->device_->set(dim, Dtype(0),
                bottom[i]->mutable_gpu_diff() + data_offset_bottom);
          } else {  // this data was been forwarded
            data_offset_top = next_to_backward_offset * dim;
            ++next_to_backward_offset;  // point to next forwarded item index
            this->device_->template copy<Dtype>(dim,
                            top[i]->mutable_gpu_diff() + data_offset_top,
                            bottom[i]->mutable_gpu_diff() + data_offset_bottom);
          }
        }
      }
    }
  }
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(FilterLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(FilterLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(FilterLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(FilterLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(FilterLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(FilterLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
