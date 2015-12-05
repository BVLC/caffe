#include <algorithm>
#include <vector>

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
#include "caffe/common_layers.hpp"
=======
#include "caffe/layers/slice_layer.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/layers/slice_layer.hpp"
>>>>>>> BVLC/master
#include "caffe/util/math_functions.hpp"
=======
=======
>>>>>>> BVLC/device-abstraction
<<<<<<< HEAD
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
=======
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> caffe
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> device-abstraction
=======
<<<<<<< HEAD
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod/caffe-merge
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/device.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> master
=======
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge

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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
  if (slice_point_.size() != 0) {
    CHECK_EQ(slice_point_.size(), top.size() - 1);
    CHECK_LE(top.size(), bottom_slice_axis);
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  if (slice_point_.size() != 0) {
    CHECK_EQ(slice_point_.size(), top.size() - 1);
    CHECK_LE(top.size(), bottom_slice_axis);
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
  count_ = 0;
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (slice_point_.size() != 0) {
    CHECK_EQ(slice_point_.size(), top.size() - 1);
    if (slice_dim_ == 0) {
      CHECK_LE(top.size(), num_);
    } else {
      CHECK_LE(top.size(), channels_);
    }
>>>>>>> origin/BVLC/parallel
=======
  if (slice_point_.size() != 0) {
    CHECK_EQ(slice_point_.size(), top.size() - 1);
    CHECK_LE(top.size(), bottom_slice_axis);
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  if (slice_point_.size() != 0) {
    CHECK_EQ(slice_point_.size(), top.size() - 1);
    CHECK_LE(top.size(), bottom_slice_axis);
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
    int prev = 0;
    vector<int> slices;
    for (int i = 0; i < slice_point_.size(); ++i) {
      CHECK_GT(slice_point_[i], prev);
      slices.push_back(slice_point_[i] - prev);
      prev = slice_point_[i];
    }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    slices.push_back(bottom_slice_axis - prev);
    for (int i = 0; i < top.size(); ++i) {
      top_shape[slice_axis_] = slices[i];
      top[i]->Reshape(top_shape);
      count += top[i]->count();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
    }
  } else {
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
    }
  } else {
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
    }
  } else {
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
    }
  } else {
>>>>>>> pod-caffe-pod.hpp-merge
=======
    }
  } else {
>>>>>>> pod/caffe-merge
    CHECK_EQ(bottom_slice_axis % top.size(), 0)
        << "Number of top blobs (" << top.size() << ") should evenly "
        << "divide input slice axis (" << bottom_slice_axis << ")";
    top_shape[slice_axis_] = bottom_slice_axis / top.size();
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(top_shape);
      count += top[i]->count();
=======
    if (slice_dim_ == 0) {
      slices.push_back(num_ - prev);
      for (int i = 0; i < top.size(); ++i) {
        top[i]->Reshape(slices[i], channels_, height_, width_);
        count_ += top[i]->count();
      }
    } else {
      slices.push_back(channels_ - prev);
      for (int i = 0; i < top.size(); ++i) {
        top[i]->Reshape(num_, slices[i], height_, width_);
        count_ += top[i]->count();
<<<<<<< HEAD
      }
    }
  } else {
    if (slice_dim_ == 0) {
      CHECK_EQ(num_ % top.size(), 0)
          << "Number of top blobs (" << top.size() << ") "
<<<<<<< HEAD
          << "should evenly divide input num ( " << num_ << ")";
      num_ = num_ / top.size();
    } else {
      CHECK_EQ(channels_ % top.size(), 0)
          << "Number of top blobs (" << top.size() << ") "
          << "should evenly divide input channels ( " << channels_ << ")";
      channels_ = channels_ / top.size();
<<<<<<< HEAD
<<<<<<< HEAD
    }
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(num_, channels_, height_, width_);
      count_ += top[i]->count();
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
    }
=======
    }
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(num_, channels_, height_, width_);
      count_ += top[i]->count();
>>>>>>> origin/BVLC/parallel
=======
    }
>>>>>>> pod/caffe-merge
  } else {
=======
    }
  } else {
>>>>>>> pod-caffe-pod.hpp-merge
    CHECK_EQ(bottom_slice_axis % top.size(), 0)
        << "Number of top blobs (" << top.size() << ") should evenly "
        << "divide input slice axis (" << bottom_slice_axis << ")";
    top_shape[slice_axis_] = bottom_slice_axis / top.size();
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(top_shape);
      count += top[i]->count();
<<<<<<< HEAD
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
    }
  } else {
>>>>>>> pod-caffe-pod.hpp-merge
=======
    }
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(num_, channels_, height_, width_);
      count_ += top[i]->count();
>>>>>>> origin/BVLC/parallel
=======
    }
  } else {
>>>>>>> pod/caffe-merge
    CHECK_EQ(bottom_slice_axis % top.size(), 0)
        << "Number of top blobs (" << top.size() << ") should evenly "
        << "divide input slice axis (" << bottom_slice_axis << ")";
    top_shape[slice_axis_] = bottom_slice_axis / top.size();
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(top_shape);
      count += top[i]->count();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
    if (slice_dim_ == 0) {
      slices.push_back(num_ - prev);
      for (int i = 0; i < top.size(); ++i) {
        top[i]->Reshape(slices[i], channels_, height_, width_);
        count_ += top[i]->count();
      }
    } else {
      slices.push_back(channels_ - prev);
      for (int i = 0; i < top.size(); ++i) {
        top[i]->Reshape(num_, slices[i], height_, width_);
        count_ += top[i]->count();
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
      }
    }
  } else {
    if (slice_dim_ == 0) {
      CHECK_EQ(num_ % top.size(), 0)
          << "Number of top blobs (" << top.size() << ") "
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
          << "should evenly divide input num ( " << num_ << ")";
      num_ = num_ / top.size();
    } else {
      CHECK_EQ(channels_ % top.size(), 0)
          << "Number of top blobs (" << top.size() << ") "
          << "should evenly divide input channels ( " << channels_ << ")";
      channels_ = channels_ / top.size();
<<<<<<< HEAD
    }
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(num_, channels_, height_, width_);
      count_ += top[i]->count();
>>>>>>> origin/BVLC/parallel
=======
    }
  } else {
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
    CHECK_EQ(bottom_slice_axis % top.size(), 0)
        << "Number of top blobs (" << top.size() << ") should evenly "
        << "divide input slice axis (" << bottom_slice_axis << ")";
    top_shape[slice_axis_] = bottom_slice_axis / top.size();
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(top_shape);
      count += top[i]->count();
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    }
  } else {
>>>>>>> device-abstraction
=======
    }
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(num_, channels_, height_, width_);
      count_ += top[i]->count();
>>>>>>> origin/BVLC/parallel
=======
    }
  } else {
>>>>>>> pod/caffe-merge
    CHECK_EQ(bottom_slice_axis % top.size(), 0)
        << "Number of top blobs (" << top.size() << ") should evenly "
        << "divide input slice axis (" << bottom_slice_axis << ")";
    top_shape[slice_axis_] = bottom_slice_axis / top.size();
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(top_shape);
      count += top[i]->count();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
    }
  }
  CHECK_EQ(count, bottom[0]->count());
  if (top.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
  CHECK_EQ(count, bottom[0]->count());
  if (top.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
}

template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
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
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
Dtype SliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
=======
void SliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
>>>>>>> origin/BVLC/parallel
  const Dtype* bottom_data = bottom[0]->mutable_cpu_data();
  if (slice_dim_ == 0) {
    int offset_num = 0;
    for (int i = 0; i < top.size(); ++i) {
      Blob<Dtype>* blob = top[i];
      Dtype* top_data = blob->mutable_cpu_data();
      GetDevice<Dtype>(Caffe::CPU)->copy(blob->count(),
          bottom_data + bottom[0]->offset(offset_num), top_data);
      offset_num += blob->num();
    }
  } else if (slice_dim_ == 1) {
    int offset_channel = 0;
    for (int i = 0; i < top.size(); ++i) {
      Blob<Dtype>* blob = top[i];
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
<<<<<<< HEAD
  return Dtype(0.);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
void SliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) { return; }
=======
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
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
>>>>>>> BVLC/master
=======
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
>>>>>>> BVLC/master
=======
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
>>>>>>> BVLC/master
=======
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
>>>>>>> master
=======
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
>>>>>>> caffe
=======
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
>>>>>>> master
=======
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
>>>>>>> master
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> device-abstraction
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
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
>>>>>>> BVLC/master
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
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
>>>>>>> master
=======
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
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
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
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
}

template <typename Dtype>
void SliceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0] || top.size() == 1) { return; }
>>>>>>> pod/device/blob.hpp
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
>>>>>>> BVLC/master
=======
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
>>>>>>> BVLC/master
=======
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
>>>>>>> BVLC/master
=======
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
>>>>>>> master
=======
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
>>>>>>> caffe
=======
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
>>>>>>> master
=======
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
>>>>>>> master
=======
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
>>>>>>> BVLC/master
=======
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
>>>>>>> master
=======
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
=======
=======
    if (slice_dim_ == 0) {
      slices.push_back(num_ - prev);
      for (int i = 0; i < top.size(); ++i) {
        top[i]->Reshape(slices[i], channels_, height_, width_);
        count_ += top[i]->count();
      }
    } else {
      slices.push_back(channels_ - prev);
      for (int i = 0; i < top.size(); ++i) {
        top[i]->Reshape(num_, slices[i], height_, width_);
        count_ += top[i]->count();
      }
    }
=======
      }
    }
>>>>>>> pod-caffe-pod.hpp-merge
  } else {
    if (slice_dim_ == 0) {
      CHECK_EQ(num_ % top.size(), 0)
          << "Number of top blobs (" << top.size() << ") "
          << "should evenly divide input num ( " << num_ << ")";
      num_ = num_ / top.size();
    } else {
      CHECK_EQ(channels_ % top.size(), 0)
          << "Number of top blobs (" << top.size() << ") "
          << "should evenly divide input channels ( " << channels_ << ")";
      channels_ = channels_ / top.size();
    }
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(num_, channels_, height_, width_);
      count_ += top[i]->count();
>>>>>>> origin/BVLC/parallel
=======
    }
  } else {
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
    CHECK_EQ(bottom_slice_axis % top.size(), 0)
        << "Number of top blobs (" << top.size() << ") should evenly "
        << "divide input slice axis (" << bottom_slice_axis << ")";
    top_shape[slice_axis_] = bottom_slice_axis / top.size();
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(top_shape);
      count += top[i]->count();
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    }
    offset_slice_axis += top_slice_axis;
  }
<<<<<<< HEAD
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
=======
  CHECK_EQ(count, bottom[0]->count());
  if (top.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
<<<<<<< HEAD
=======
  CHECK_EQ(count, bottom[0]->count());
  if (top.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
>>>>>>> pod-caffe-pod.hpp-merge
}

template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
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
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
Dtype SliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
=======
void SliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
>>>>>>> origin/BVLC/parallel
  const Dtype* bottom_data = bottom[0]->mutable_cpu_data();
  if (slice_dim_ == 0) {
    int offset_num = 0;
    for (int i = 0; i < top.size(); ++i) {
      Blob<Dtype>* blob = top[i];
      Dtype* top_data = blob->mutable_cpu_data();
      GetDevice<Dtype>(Caffe::CPU)->copy(blob->count(),
          bottom_data + bottom[0]->offset(offset_num), top_data);
      offset_num += blob->num();
    }
  } else if (slice_dim_ == 1) {
    int offset_channel = 0;
    for (int i = 0; i < top.size(); ++i) {
      Blob<Dtype>* blob = top[i];
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
<<<<<<< HEAD
  return Dtype(0.);
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
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
>>>>>>> BVLC/master
=======
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
>>>>>>> BVLC/master
=======
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
>>>>>>> BVLC/master
=======
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
>>>>>>> master
=======
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
>>>>>>> caffe
=======
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
>>>>>>> master
=======
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
>>>>>>> master
=======
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
>>>>>>> BVLC/master
=======
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
>>>>>>> master
=======
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
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
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
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
}

template <typename Dtype>
void SliceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
<<<<<<< HEAD
=======
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
=======
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
>>>>>>> origin/BVLC/parallel
  if (!propagate_down[0]) { return; }
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (slice_dim_ == 0) {
    int offset_num = 0;
    for (int i = 0; i < top.size(); ++i) {
      Blob<Dtype>* blob = top[i];
      const Dtype* top_diff = blob->cpu_diff();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
      GetDevice<Dtype>(Caffe::CPU)->copy(blob->count(), top_diff,
          bottom_diff + (*bottom)[0]->offset(offset_num));
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
      GetDevice<Dtype>(Caffe::CPU)->copy(blob->count(), top_diff,
          bottom_diff + (*bottom)[0]->offset(offset_num));
=======
      caffe_copy(blob->count(), top_diff,
                 bottom_diff + bottom[0]->offset(offset_num));
>>>>>>> origin/BVLC/parallel
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
      caffe_copy(blob->count(), top_diff,
                 bottom_diff + bottom[0]->offset(offset_num));
>>>>>>> origin/BVLC/parallel
>>>>>>> pod-caffe-pod.hpp-merge
=======
      caffe_copy(blob->count(), top_diff,
                 bottom_diff + bottom[0]->offset(offset_num));
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/caffe-merge
=======
      caffe_copy(blob->count(), top_diff,
                 bottom_diff + bottom[0]->offset(offset_num));
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
      GetDevice<Dtype>(Caffe::CPU)->copy(blob->count(), top_diff,
          bottom_diff + (*bottom)[0]->offset(offset_num));
>>>>>>> BVLC/device-abstraction
=======
      GetDevice<Dtype>(Caffe::CPU)->copy(blob->count(), top_diff,
          bottom_diff + (*bottom)[0]->offset(offset_num));
>>>>>>> device-abstraction
=======
=======
      GetDevice<Dtype>(Caffe::CPU)->copy(blob->count(), top_diff,
          bottom_diff + (*bottom)[0]->offset(offset_num));
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
      GetDevice<Dtype>(Caffe::CPU)->copy(blob->count(), top_diff,
          bottom_diff + (*bottom)[0]->offset(offset_num));
=======
      caffe_copy(blob->count(), top_diff,
                 bottom_diff + bottom[0]->offset(offset_num));
>>>>>>> origin/BVLC/parallel
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
      GetDevice<Dtype>(Caffe::CPU)->copy(blob->count(), top_diff,
          bottom_diff + (*bottom)[0]->offset(offset_num));
=======
      caffe_copy(blob->count(), top_diff,
                 bottom_diff + bottom[0]->offset(offset_num));
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
      GetDevice<Dtype>(Caffe::CPU)->copy(blob->count(), top_diff,
          bottom_diff + (*bottom)[0]->offset(offset_num));
=======
      caffe_copy(blob->count(), top_diff,
                 bottom_diff + bottom[0]->offset(offset_num));
>>>>>>> origin/BVLC/parallel
>>>>>>> pod-caffe-pod.hpp-merge
      offset_num += blob->num();
    }
  } else if (slice_dim_ == 1) {
    int offset_channel = 0;
    for (int i = 0; i < top.size(); ++i) {
      Blob<Dtype>* blob = top[i];
      const Dtype* top_diff = blob->cpu_diff();
      const int num_elem = blob->channels() * blob->height() * blob->width();
      for (int n = 0; n < num_; ++n) {
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
        GetDevice<Dtype>(Caffe::CPU)->copy(num_elem, top_diff + blob->offset(n),
            bottom_diff + (*bottom)[0]->offset(n, offset_channel));
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
      }
      offset_channel += blob->channels();
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
        GetDevice<Dtype>(Caffe::CPU)->copy(num_elem, top_diff + blob->offset(n),
            bottom_diff + (*bottom)[0]->offset(n, offset_channel));
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
        caffe_copy(num_elem, top_diff + blob->offset(n),
                   bottom_diff + bottom[0]->offset(n, offset_channel));
>>>>>>> origin/BVLC/parallel
=======
        GetDevice<Dtype>(Caffe::CPU)->copy(num_elem, top_diff + blob->offset(n),
            bottom_diff + (*bottom)[0]->offset(n, offset_channel));
>>>>>>> BVLC/device-abstraction
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
        GetDevice<Dtype>(Caffe::CPU)->copy(num_elem, top_diff + blob->offset(n),
            bottom_diff + (*bottom)[0]->offset(n, offset_channel));
=======
        caffe_copy(num_elem, top_diff + blob->offset(n),
                   bottom_diff + bottom[0]->offset(n, offset_channel));
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
        caffe_copy(num_elem, top_diff + blob->offset(n),
                   bottom_diff + bottom[0]->offset(n, offset_channel));
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
      }
      offset_channel += blob->channels();
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
        GetDevice<Dtype>(Caffe::CPU)->copy(num_elem, top_diff + blob->offset(n),
            bottom_diff + (*bottom)[0]->offset(n, offset_channel));
      }
      offset_channel += blob->channels();
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/master
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
    }
    offset_slice_axis += top_slice_axis;
  }
}

#ifdef CPU_ONLY
STUB_GPU(SliceLayer);
#endif

INSTANTIATE_CLASS(SliceLayer);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
REGISTER_LAYER_CLASS(Slice);

=======
REGISTER_LAYER_CLASS(SLICE, SliceLayer);
>>>>>>> origin/BVLC/parallel
}  // namespace caffe
