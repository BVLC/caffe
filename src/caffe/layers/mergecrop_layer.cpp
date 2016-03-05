#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/mergecrop_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void MergeCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  // By default, forward both a and b
  forward_.push_back(1);
  forward_.push_back(1);

  // By default, backward a and do not backward b
  backward_.push_back(1);
  backward_.push_back(0);

  op_ = MergeCropParameter_MergeOp_STACK;

  if (this->layer_param_.has_mergecrop_param()) {
    MergeCropParameter mergecrop_param = this->layer_param_.mergecrop_param();
    for (int_tp i = 0; i < mergecrop_param.forward_size(); ++i) {
      forward_[i] = mergecrop_param.forward(i);
    }
    for (int_tp i = 0; i < mergecrop_param.backward_size(); ++i) {
      backward_[i] = mergecrop_param.backward(i);
    }
    op_ = mergecrop_param.operation();
  }

  Reshape(bottom, top);
}

template<typename Dtype>
void MergeCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  // Same number of batches required
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));

  int_tp channels = 0;
  if (op_ == MergeCropParameter_MergeOp_STACK) {
    // All channels of both inputs are copied
    channels = bottom[0]->shape(1) + bottom[1]->shape(1);
  } else {
    // Same number of feature maps required
    CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
    channels = bottom[0]->shape(1);
  }

  // Spatial of the smaller input, which should be input 0
  vector<int_tp> top_shape = bottom[0]->shape();
  top_shape[1] = channels;

  top[0]->Reshape(top_shape);

  shape_a_.Reshape(1, 1, 1, top_shape.size() - 2);
  shape_b_.Reshape(1, 1, 1, top_shape.size() - 2);

  int_tp* shape_a_data = shape_a_.mutable_cpu_data();
  int_tp* shape_b_data = shape_b_.mutable_cpu_data();

  for (int_tp i = 0; i < top_shape.size() - 2; ++i) {
    shape_a_data[i] = bottom[0]->shape()[i + 2];
    shape_b_data[i] = bottom[1]->shape()[i + 2];
  }
}

template<typename Dtype>
void MergeCropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  LOG(FATAL)<< "Foward_cpu() not implemented for MergeCropLayer.";
}

template<typename Dtype>
void MergeCropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL)<< "Backward_cpu() not implemented for MergeCropLayer.";
}

#ifdef CPU_ONLY
STUB_GPU(MergeCropLayer);
#endif

INSTANTIATE_CLASS(MergeCropLayer);
REGISTER_LAYER_CLASS(MergeCrop);

}  // namespace caffe
