#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SliceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const SliceParameter& slice_param = this->layer_param_.slice_param();
  CHECK(!(slice_param.has_axis() && slice_param.has_slice_dim()))
      << "Either axis or slice_dim should be specified; not both.";
  slice_point_.clear();
  std::copy(slice_param.slice_point().begin(),
      slice_param.slice_point().end(),
      std::back_inserter(slice_point_));
}

template <typename Dtype>
void SliceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  const SliceParameter& slice_param = this->layer_param_.slice_param();
  if (slice_param.has_slice_dim()) {
    slice_axis_ = static_cast<int>(slice_param.slice_dim());
    // Don't allow negative indexing for slice_dim, a uint32 -- almost
    // certainly unintended.
    CHECK_GE(slice_axis_, 0) << "casting slice_dim from uint32 to int32 "
        << "produced negative result; slice_dim must satisfy "
        << "0 <= slice_dim < " << kMaxBlobAxes;
    CHECK_LT(slice_axis_, num_axes) << "slice_dim out of range.";
  } else {
    slice_axis_ = bottom[0]->CanonicalAxisIndex(slice_param.axis());
  }
  vector<int> top_shape = bottom[0]->shape();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  num_slices_ = bottom[0]->count(0, slice_axis_);
  slice_size_ = bottom[0]->count(slice_axis_ + 1);
  int count = 0;
  if (slice_point_.size() != 0) {
    CHECK_EQ(slice_point_.size(), top.size() - 1);
    CHECK_LE(top.size(), bottom_slice_axis);
    int prev = 0;
    vector<int> slices;
    for (int i = 0; i < slice_point_.size(); ++i) {
      CHECK_GT(slice_point_[i], prev);
      slices.push_back(slice_point_[i] - prev);
      prev = slice_point_[i];
    }
    slices.push_back(bottom_slice_axis - prev);
    for (int i = 0; i < top.size(); ++i) {
      top_shape[slice_axis_] = slices[i];
      top[i]->Reshape(top_shape);
      count += top[i]->count();
    }
  } else {
    CHECK_EQ(bottom_slice_axis % top.size(), 0)
        << "Number of top blobs (" << top.size() << ") should evenly "
        << "divide input slice axis (" << bottom_slice_axis << ")";
    top_shape[slice_axis_] = bottom_slice_axis / top.size();
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(top_shape);
      count += top[i]->count();
    }
  }
  CHECK_EQ(count, bottom[0]->count());
}

template <typename Dtype>
void SliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int offset_slice_axis = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  for (int i = 0; i < top.size(); ++i) {
    Dtype* top_data = top[i]->mutable_cpu_data();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    for (int n = 0; n < num_slices_; ++n) {
      const int top_offset = n * top_slice_axis * slice_size_;
      const int bottom_offset =
          (n * bottom_slice_axis + offset_slice_axis) * slice_size_;
      caffe_copy(top_slice_axis * slice_size_,
          bottom_data + bottom_offset, top_data + top_offset);
    }
    offset_slice_axis += top_slice_axis;
  }
}

template <typename Dtype>
void SliceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  int offset_slice_axis = 0;
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    for (int n = 0; n < num_slices_; ++n) {
      const int top_offset = n * top_slice_axis * slice_size_;
      const int bottom_offset =
          (n * bottom_slice_axis + offset_slice_axis) * slice_size_;
      caffe_copy(top_slice_axis * slice_size_,
          top_diff + top_offset, bottom_diff + bottom_offset);
    }
    offset_slice_axis += top_slice_axis;
  }
}

#ifdef CPU_ONLY
STUB_GPU(SliceLayer);
#endif

INSTANTIATE_CLASS(SliceLayer);
REGISTER_LAYER_CLASS(Slice);

}  // namespace caffe
