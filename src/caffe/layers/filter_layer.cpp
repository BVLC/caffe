#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void FilterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), bottom.size()-1) <<
      "Top.size() should be equal to bottom.size() - 1";
  const FilterParameter& filter_param = this->layer_param_.filter_param();
  first_reshape_ = true;
  need_back_prop_.clear();
  std::copy(filter_param.need_back_prop().begin(),
      filter_param.need_back_prop().end(),
      std::back_inserter(need_back_prop_));
  CHECK_NE(0, need_back_prop_.size()) <<
      "need_back_prop param needs to be specified";
  CHECK_EQ(top.size(), need_back_prop_.size()) <<
      "need_back_prop.size() needs to be equal to top.size()";
}

template <typename Dtype>
void FilterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom[0...k-1] are the blobs to filter
  // bottom[last] is the "selector_blob"
  int selector_index = bottom.size() - 1;
  CHECK_EQ(bottom[selector_index]->channels(), 1) <<
        "Selector blob (bottom[last]) must have channels == 1";
  CHECK_EQ(bottom[selector_index]->width(), 1) <<
        "Selector blob (bottom[last]) must have width == 1";
  CHECK_EQ(bottom[selector_index]->height(), 1) <<
        "Selector blob (bottom[last]) must have height == 1";
  for (int i = 0; i < bottom.size()-1; i++) {
    CHECK_EQ(bottom[selector_index]->num(), bottom[i]->num()) <<
        "Each bottom should have the same dimension as bottom[last]";
  }

  const Dtype* bottom_data_selector = bottom[selector_index]->cpu_data();
  indices_to_forward_.clear();

  // look for non-zero elements in bottom[0]. Items of each bottom that
  // have the same index as the items in bottom[0] with value == non-zero
  // will be forwarded
  for (int item_id = 0; item_id < bottom[selector_index]->num(); ++item_id) {
    // we don't need an offset because item size == 1
    const Dtype* tmp_data_selector = bottom_data_selector + item_id;
    if (*tmp_data_selector) {
      indices_to_forward_.push_back(item_id);
    }
  }
  // only filtered items will be forwarded
  int new_tops_num = indices_to_forward_.size();
  // init
  if (first_reshape_) {
    new_tops_num = bottom[0]->num();
    first_reshape_ = false;
  }
  for (int t = 0; t < top.size(); t++) {
    top[t]->Reshape(new_tops_num, bottom[t]->channels(),
        bottom[t]->height(), bottom[t]->width());
  }
}

template <typename Dtype>
void FilterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int new_tops_num = indices_to_forward_.size();
  // forward all filtered items for all bottoms but the Selector (bottom[last])
  for (int t = 0; t < top.size(); t++) {
    const Dtype* bottom_data = bottom[t]->cpu_data();
    Dtype* top_data = top[t]->mutable_cpu_data();
    int dim = bottom[t]->count() / bottom[t]->num();
    for (int n = 0; n < new_tops_num; n++) {
      int data_offset_top = top[t]->offset(n);
      int data_offset_bottom =  bottom[t]->offset(indices_to_forward_[n]);
      caffe_copy(dim, bottom_data + data_offset_bottom,
          top_data + data_offset_top);
    }
  }
}

template <typename Dtype>
void FilterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < top.size(); i++) {
    // bottom[last] is the selector and never needs backpropagation
    if (propagate_down[i] && need_back_prop_[i]) {
      const int dim = top[i]->count() / top[i]->num();
      int next_to_backward_offset = 0;
      int batch_offset = 0;
      int data_offset_bottom = 0;
      int data_offset_top = 0;
      for (int n = 0; n < bottom[i]->num(); n++) {
        if (next_to_backward_offset >= indices_to_forward_.size()) {
          // we already visited all items that were been forwarded, so
          // just set to zero remaining ones
          data_offset_bottom = top[i]->offset(n);
          caffe_set(dim, Dtype(0),
              bottom[i]->mutable_cpu_diff() + data_offset_bottom);
        } else {
          batch_offset = indices_to_forward_[next_to_backward_offset];
          data_offset_bottom = top[i]->offset(n);
          if (n != batch_offset) {  // this data was not been forwarded
            caffe_set(dim, Dtype(0),
                bottom[i]->mutable_cpu_diff() + data_offset_bottom);
          } else {  // this data was been forwarded
            data_offset_top = top[i]->offset(next_to_backward_offset);
            next_to_backward_offset++;  // point to next forwarded item index
            caffe_copy(dim, top[i]->mutable_cpu_diff() + data_offset_top,
                bottom[i]->mutable_cpu_diff() + data_offset_bottom);
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FilterLayer);
#endif

INSTANTIATE_CLASS(FilterLayer);
REGISTER_LAYER_CLASS(FILTER, FilterLayer);
}  // namespace caffe
