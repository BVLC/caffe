#include <algorithm>
#include <vector>

<<<<<<< HEAD
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> BVLC/device-abstraction

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
  if (top.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
}

template <typename Dtype>
<<<<<<< HEAD
void SliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) { return; }
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
=======
Dtype SliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->mutable_cpu_data();
  if (slice_dim_ == 0) {
    int offset_num = 0;
    for (int i = 0; i < top->size(); ++i) {
      Blob<Dtype>* blob = (*top)[i];
      Dtype* top_data = blob->mutable_cpu_data();
      GetDevice<Dtype>(Caffe::CPU)->copy(blob->count(),
          bottom_data + bottom[0]->offset(offset_num), top_data);
      offset_num += blob->num();
    }
  } else if (slice_dim_ == 1) {
    int offset_channel = 0;
    for (int i = 0; i < top->size(); ++i) {
      Blob<Dtype>* blob = (*top)[i];
      Dtype* top_data = blob->mutable_cpu_data();
      const int num_elem = blob->channels() * blob->height() * blob->width();
      for (int n = 0; n < num_; ++n) {
        GetDevice<Dtype>(Caffe::CPU)->copy(num_elem,
            bottom_data + bottom[0]->offset(n, offset_channel),
            top_data + blob->offset(n));
      }
      offset_channel += blob->channels();
    }
  }  // slice_dim_ is guaranteed to be 0 or 1 by SetUp.
  return Dtype(0.);
>>>>>>> BVLC/device-abstraction
}

template <typename Dtype>
void SliceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
<<<<<<< HEAD
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0] || top.size() == 1) { return; }
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
=======
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) { return; }
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  if (slice_dim_ == 0) {
    int offset_num = 0;
    for (int i = 0; i < top.size(); ++i) {
      Blob<Dtype>* blob = top[i];
      const Dtype* top_diff = blob->cpu_diff();
      GetDevice<Dtype>(Caffe::CPU)->copy(blob->count(), top_diff,
          bottom_diff + (*bottom)[0]->offset(offset_num));
      offset_num += blob->num();
    }
  } else if (slice_dim_ == 1) {
    int offset_channel = 0;
    for (int i = 0; i < top.size(); ++i) {
      Blob<Dtype>* blob = top[i];
      const Dtype* top_diff = blob->cpu_diff();
      const int num_elem = blob->channels() * blob->height() * blob->width();
      for (int n = 0; n < num_; ++n) {
        GetDevice<Dtype>(Caffe::CPU)->copy(num_elem, top_diff + blob->offset(n),
            bottom_diff + (*bottom)[0]->offset(n, offset_channel));
      }
      offset_channel += blob->channels();
>>>>>>> BVLC/device-abstraction
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
