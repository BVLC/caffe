#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/flip_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

template <typename Dtype>
void FlipLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  flip_width_ = this->layer_param_.flip_param().flip_width();
  flip_height_ = this->layer_param_.flip_param().flip_height();
}

template <typename Dtype>
void FlipLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void FlipLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int width = bottom[0]->width();
  int height = bottom[0]->height();

  for (int n = 0; n < num; n++) {
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          top_data[(((n * channels + c) * height + h) * width) + w] = 
            bottom_data[(((n * channels + c) * height + (flip_height_ ? (height - 1 - h) : h)) * width) + (flip_width_ ? (width - 1 - w) : w)];
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(FlipLayer);
#endif

INSTANTIATE_CLASS(FlipLayer);
REGISTER_LAYER_CLASS(Flip);

}  // namespace caffe
