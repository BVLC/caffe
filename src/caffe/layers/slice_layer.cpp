#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SliceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const SliceParameter& slice_param = this->layer_param_.slice_param();
  slice_dim_ = slice_param.slice_dim();
  CHECK_GE(slice_dim_, 0);
  CHECK_LE(slice_dim_, 1) << "Can only slice num and channels";
  slice_point_.clear();
  std::copy(slice_param.slice_point().begin(),
      slice_param.slice_point().end(),
      std::back_inserter(slice_point_));
}

template <typename Dtype>
void SliceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  count_ = 0;
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (slice_point_.size() != 0) {
    CHECK_EQ(slice_point_.size(), top->size() - 1);
    if (slice_dim_ == 0) {
      CHECK_LE(top->size(), num_);
    } else {
      CHECK_LE(top->size(), channels_);
    }
    int prev = 0;
    vector<int> slices;
    for (int i = 0; i < slice_point_.size(); ++i) {
      CHECK_GT(slice_point_[i], prev);
      slices.push_back(slice_point_[i] - prev);
      prev = slice_point_[i];
    }
    if (slice_dim_ == 0) {
      slices.push_back(num_ - prev);
      for (int i = 0; i < top->size(); ++i) {
        (*top)[i]->Reshape(slices[i], channels_, height_, width_);
         count_ += (*top)[i]->count();
      }
    } else {
      slices.push_back(channels_ - prev);
      for (int i = 0; i < top->size(); ++i) {
        (*top)[i]->Reshape(num_, slices[i], height_, width_);
         count_ += (*top)[i]->count();
      }
    }
  } else {
    if (slice_dim_ == 0) {
      CHECK_EQ(num_ % top->size(), 0)
          << "Number of top blobs (" << top->size() << ") "
          << "should evenly divide input num ( " << num_ << ")";
      num_ = num_ / top->size();
    } else {
      CHECK_EQ(channels_ % top->size(), 0)
          << "Number of top blobs (" << top->size() << ") "
          << "should evenly divide input channels ( " << channels_ << ")";
      channels_ = channels_ / top->size();
    }
    for (int i = 0; i < top->size(); ++i) {
      (*top)[i]->Reshape(num_, channels_, height_, width_);
      count_ += (*top)[i]->count();
    }
  }
  CHECK_EQ(count_, bottom[0]->count());
}

template <typename Dtype>
void SliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->mutable_cpu_data();
  if (slice_dim_ == 0) {
    int offset_num = 0;
    for (int i = 0; i < top->size(); ++i) {
      Blob<Dtype>* blob = (*top)[i];
      Dtype* top_data = blob->mutable_cpu_data();
      caffe_copy(blob->count(), bottom_data + bottom[0]->offset(offset_num),
                 top_data);
      offset_num += blob->num();
    }
  } else if (slice_dim_ == 1) {
    int offset_channel = 0;
    for (int i = 0; i < top->size(); ++i) {
      Blob<Dtype>* blob = (*top)[i];
      Dtype* top_data = blob->mutable_cpu_data();
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
void SliceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) { return; }
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  if (slice_dim_ == 0) {
    int offset_num = 0;
    for (int i = 0; i < top.size(); ++i) {
      Blob<Dtype>* blob = top[i];
      const Dtype* top_diff = blob->cpu_diff();
      caffe_copy(blob->count(), top_diff,
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
        caffe_copy(num_elem, top_diff + blob->offset(n),
                   bottom_diff + (*bottom)[0]->offset(n, offset_channel));
      }
      offset_channel += blob->channels();
    }
  }  // slice_dim_ is guaranteed to be 0 or 1 by SetUp.
}

#ifdef CPU_ONLY
STUB_GPU(SliceLayer);
#endif

INSTANTIATE_CLASS(SliceLayer);

}  // namespace caffe
