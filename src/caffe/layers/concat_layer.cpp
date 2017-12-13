#include <vector>

#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim()))
      << "Either axis or concat_dim should be specified; not both.";
}

template <typename Dtype>
  int ConcatLayer<Dtype>::get_concat_axis(const vector<Blob<Dtype>*>& bottom) const {
    int concat_axis;
  const int num_axes = bottom[0]->num_axes();
  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  if (concat_param.has_concat_dim()) {
    concat_axis = static_cast<int>(concat_param.concat_dim());
    // Don't allow negative indexing for concat_dim, a uint32 -- almost
    // certainly unintended.
    CHECK_GE(concat_axis, 0) << "casting concat_dim from uint32 to int32 "
        << "produced negative result; concat_dim must satisfy "
        << "0 <= concat_dim < " << kMaxBlobAxes;
    CHECK_LT(concat_axis, num_axes) << "concat_dim out of range.";
  } else {
    concat_axis = bottom[0]->CanonicalAxisIndex(concat_param.axis());
  }
  return concat_axis;
}


template <typename Dtype>
void ConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Reshape_const(bottom,top);
}

template <typename Dtype>
void ConcatLayer<Dtype>::Reshape_const(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)  const {
  const int num_axes = bottom[0]->num_axes();
  // Initialize with the first blob.
  vector<int> top_shape = bottom[0]->shape();
  auto concat_axis= get_concat_axis(bottom);
  int bottom_count_sum = bottom[0]->count();
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num_axes, bottom[i]->num_axes())
        << "All inputs must have the same #axes.";
    for (int j = 0; j < num_axes; ++j) {
      if (j == concat_axis) { continue; }
      CHECK_EQ(top_shape[j], bottom[i]->shape(j))
          << "All inputs must have the same shape, except at concat_axis.";
    }
    bottom_count_sum += bottom[i]->count();
    top_shape[concat_axis] += bottom[i]->shape(concat_axis);
  }
  top[0]->Reshape(top_shape);
  CHECK_EQ(bottom_count_sum, top[0]->count());
  if (bottom.size() == 1) {
    top[0]->ShareData(*bottom[0]);
  }
}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Forward_const_cpu(bottom,top);
}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_const_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const{
  if (bottom.size() == 1) { return; }
  Dtype* top_data = top[0]->mutable_cpu_data();
  int offset_concat_axis = 0;
  auto concat_axis= get_concat_axis(bottom);
  const int top_concat_axis = top[0]->shape(concat_axis);
  int num_concats = bottom[0]->count(0, concat_axis);
  int concat_input_size = bottom[0]->count(concat_axis + 1);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis);
    for (int n = 0; n < num_concats; ++n) {
      caffe_copy(bottom_concat_axis * concat_input_size,
          bottom_data + n * bottom_concat_axis * concat_input_size,
          top_data + (n * top_concat_axis + offset_concat_axis)
              * concat_input_size);
    }
    offset_concat_axis += bottom_concat_axis;
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConcatLayer);
STUB_GPU_FORWARD_CONST(ConcatLayer,Forward_const);
#endif

INSTANTIATE_CLASS(ConcatLayer);
REGISTER_LAYER_CLASS(Concat);

}  // namespace caffe
