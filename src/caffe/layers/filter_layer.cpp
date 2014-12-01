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
  // bottom[0] is the "selector_blob"
  // bottom[1+] are the blobs to filter
  CHECK_EQ(bottom[0]->channels(), 1) <<
        "Selector blob (bottom[0]) must have channels == 1";
  CHECK_EQ(bottom[0]->width(), 1) <<
        "Selector blob (bottom[0]) must have width == 1";
  CHECK_EQ(bottom[0]->height(), 1) <<
        "Selector blob (bottom[0]) must have height == 1";
  int num_items = bottom[0]->num();
  for (int i = 1; i < bottom.size(); i++) {
    CHECK_EQ(num_items, bottom[i]->num()) <<
        "Each bottom should have the same dimension as bottom[0]";
  }

  const Dtype* bottom_data_selector = bottom[0]->cpu_data();
  indices_to_forward_.clear();

  // look for non-zero elements in bottom[0]. Items of each bottom that
  // have the same index as the items in bottom[0] with value == non-zero
  // will be forwarded
  for (int item_id = 0; item_id < num_items; ++item_id) {
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
    new_tops_num = bottom[1]->num();
    first_reshape_ = false;
  }

  for (int t = 0; t < top.size(); t++) {
    top[t]->Reshape(new_tops_num,
        bottom[t+1]->channels(),
        bottom[t+1]->height(),
        bottom[t+1]->width());
  }
}

template <typename Dtype>
void FilterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int new_tops_num = indices_to_forward_.size();
  // forward all filtered items for all bottoms but the Selector (bottom[0])
  for (int b = 1; b < bottom.size(); b++) {
    const Dtype* bottom_data = bottom[b]->cpu_data();
    Dtype* top_data = top[b-1]->mutable_cpu_data();
    int dim = bottom[b]->count() / bottom[b]->num();

    for (int n = 0; n < new_tops_num; n++) {
      int offset = indices_to_forward_[n];
      int data_offset_top = top[b-1]->offset(n);
      int data_offset_bottom =  bottom[b]->offset(indices_to_forward_[n]);

      caffe_copy(dim, bottom_data+data_offset_bottom,
          top_data+data_offset_top);
    }
  }
}

template <typename Dtype>
void FilterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 1; i < propagate_down.size(); i++) {
    // bottom[0] is the selector and never needs backpropagation
    // so we can start from i = 1 and index each top with i-1
    if (propagate_down[i] && need_back_prop_[i-1]) {
      const int dim = top[i-1]->count() / top[i-1]->num();
      int next_to_backward_offset = 0;
      int batch_offset = 0;
      int data_offset_bottom = 0;
      int data_offset_top = 0;
      for (int n = 0; n < bottom[i]->num(); n++) {
        if (next_to_backward_offset >= indices_to_forward_.size()) {
          // we already visited all items that were been forwarded, so
          // just set to zero remaining ones
          data_offset_bottom = top[i-1]->offset(n);
          caffe_set(dim, Dtype(0),
              bottom[i]->mutable_cpu_diff() + data_offset_bottom);
        } else {
          batch_offset = indices_to_forward_[next_to_backward_offset];
          data_offset_bottom = top[i-1]->offset(n);
          if (n != batch_offset) {  // this data was not been forwarded
            caffe_set(dim, Dtype(0),
                bottom[i]->mutable_cpu_diff() + data_offset_bottom);
          } else {  // this data was been forwarded
            data_offset_top = top[i-1]->offset(next_to_backward_offset);
            next_to_backward_offset++;  // point to next forwarded item index
            caffe_copy(dim, top[i-1]->mutable_cpu_diff() + data_offset_top,
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
