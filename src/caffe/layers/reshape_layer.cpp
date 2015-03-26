#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  inferred_axis_ = -1;
  copy_axes_.clear();
  const BlobShape& top_blob_shape = this->layer_param_.reshape_param().shape();
  const int top_num_axes = top_blob_shape.dim_size();
  top_shape_.resize(top_num_axes);
  constant_count_ = 1;
  for (int i = 0; i < top_num_axes; ++i) {
    top_shape_[i] = top_blob_shape.dim(i);
    if (top_shape_[i] == 0) {
      copy_axes_.push_back(i);
    } else if (top_shape_[i] == -1) {
      CHECK_EQ(inferred_axis_, -1) << "new shape contains multiple "
          << "-1 dims; at most a single (1) value of -1 may be specified";
      inferred_axis_ = i;
    } else {
      constant_count_ *= top_shape_[i];
    }
  }
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < copy_axes_.size(); ++i) {
    const int copy_axis_index = copy_axes_[i];
    CHECK_GT(bottom[0]->num_axes(), copy_axis_index) << "new shape contains "
        << "a 0, but there is no corresponding bottom axis to copy";
    top_shape_[copy_axis_index] = bottom[0]->shape(copy_axis_index);
  }
  if (inferred_axis_ >= 0) {
    // A -1 dim was specified; infer the correct dimension by computing the
    // product of the other dimensions.
    int explicit_count = constant_count_;
    for (int i = 0; i < copy_axes_.size(); ++i) {
      const int copy_axis_index = copy_axes_[i];
      explicit_count *= top_shape_[copy_axis_index];
    }
    CHECK_EQ(0, bottom[0]->count() % explicit_count) << "bottom count ("
        << bottom[0]->count() << ") must be divisible by the product of "
        << "the specified dimensions (" << explicit_count << ")";
    const int inferred_dim = bottom[0]->count() / explicit_count;
    top_shape_[inferred_axis_] = inferred_dim;
  }
  top[0]->Reshape(top_shape_);
  CHECK_EQ(top[0]->count(), bottom[0]->count())
      << "output count must match input count";
  top[0]->ShareData(*bottom[0]);
  top[0]->ShareDiff(*bottom[0]);
}

INSTANTIATE_CLASS(ReshapeLayer);
REGISTER_LAYER_CLASS(Reshape);

}  // namespace caffe
