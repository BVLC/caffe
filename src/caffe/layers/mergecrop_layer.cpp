#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void MergeCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  std::vector<int> forward_shape(1);
  forward_shape[0] = 2;

  std::vector<int> backward_shape(1);
  backward_shape[0] = 2;

  forward.Reshape(forward_shape);
  backward.Reshape(backward_shape);

  int* forward_data = forward.mutable_cpu_data();
  int* backward_data = backward.mutable_cpu_data();

  // By default, forward both a and b
  forward_data[0] = 1;
  forward_data[1] = 1;

  // By default, backward a and do not backward b
  backward_data[0] = 1;
  backward_data[1] = 0;


  if (this->layer_param_.has_mergecrop_param()) {
    MergeCropParameter mergecrop_param = this->layer_param_.mergecrop_param();
    for (int i = 0; i < mergecrop_param.forward_size(); ++i) {
      forward_data[i] = mergecrop_param.forward(i);
    }
    for (int i = 0; i < mergecrop_param.backward_size(); ++i) {
      backward_data[i] = mergecrop_param.backward(i);
    }
  }

  Reshape(bottom, top);
}

template<typename Dtype>
void MergeCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  // Same number of batches requires
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  int num = bottom[0]->num();

  // All channels of both inputs are copied
  int channels = bottom[0]->channels() + bottom[1]->channels();

  // Width and height of the smaller input, which should be input 0
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  top[0]->Reshape(num, channels, height, width);
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
