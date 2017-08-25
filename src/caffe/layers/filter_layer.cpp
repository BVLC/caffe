#include <vector>

#include "caffe/layers/filter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FilterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), bottom.size() - 1);
  first_reshape_ = true;
}

template <typename Dtype>
void FilterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom[0...k-1] are the blobs to filter
  // bottom[last] is the "selector_blob"
  int selector_index = bottom.size() - 1;
  for (int i = 1; i < bottom[selector_index]->num_axes(); ++i) {
    CHECK_EQ(bottom[selector_index]->shape(i), 1)
        << "Selector blob dimensions must be singletons (1), except the first";
  }
  for (int i = 0; i < bottom.size() - 1; ++i) {
    CHECK_EQ(bottom[selector_index]->shape(0), bottom[i]->shape(0)) <<
        "Each bottom should have the same 0th dimension as the selector blob";
  }

  const Dtype* bottom_data_selector = bottom[selector_index]->cpu_data();
  indices_to_forward_.clear();

  // look for non-zero elements in bottom[0]. Items of each bottom that
  // have the same index as the items in bottom[0] with value == non-zero
  // will be forwarded
  for (int item_id = 0; item_id < bottom[selector_index]->shape(0); ++item_id) {
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
    new_tops_num = bottom[0]->shape(0);
    first_reshape_ = false;
  }
  for (int t = 0; t < top.size(); ++t) {
    int num_axes = bottom[t]->num_axes();
    vector<int> shape_top(num_axes);
    shape_top[0] = new_tops_num;
    for (int ts = 1; ts < num_axes; ++ts)
      shape_top[ts] = bottom[t]->shape(ts);
    top[t]->Reshape(shape_top);
  }
}

template <typename Dtype>
void FilterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int new_tops_num = indices_to_forward_.size();
  // forward all filtered items for all bottoms but the Selector (bottom[last])
  for (int t = 0; t < top.size(); ++t) {
    const Dtype* bottom_data = bottom[t]->cpu_data();
    Dtype* top_data = top[t]->mutable_cpu_data();
    int dim = bottom[t]->count() / bottom[t]->shape(0);
    for (int n = 0; n < new_tops_num; ++n) {
      int data_offset_top = n * dim;
      int data_offset_bottom = indices_to_forward_[n] * bottom[t]->count(1);
      caffe_copy(dim, bottom_data + data_offset_bottom,
          top_data + data_offset_top);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FilterLayer);
#endif

INSTANTIATE_CLASS(FilterLayer);
REGISTER_LAYER_CLASS(Filter);

}  // namespace caffe
