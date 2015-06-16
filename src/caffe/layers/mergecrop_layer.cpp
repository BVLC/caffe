#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void MergeCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  // Nothing to do here, other than the reshaping
  Reshape(bottom, top);
}

template<typename Dtype>
void MergeCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {

  // Same number of batches requires
  CHECK_EQ(bottom[0]->num(),bottom[1]->num());
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
