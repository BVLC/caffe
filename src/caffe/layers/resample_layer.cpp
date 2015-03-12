#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void ResampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ResampleParameter resample_param = this->layer_param_.resample_param();
  CHECK(resample_param.has_num_output())
      << "Needs number of outputs.";
  num_output_ = resample_param.num_output();
  CHECK_GT(num_output_, 0) << "num_output must be greater than zero.";
  if (resample_param.has_image_w()) {
    image_w_ = resample_param.image_w();
    image_h_ = resample_param.image_h();
  }
}

template <typename Dtype>
void ResampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  // For windowed case.
  if (bottom.size() > 1) {
    top[0]->Reshape(bottom[0]->num(), channels_, num_output_, bottom[1]->height());
  } else {
    top[0]->Reshape(bottom[0]->num(), channels_, num_output_, 1);
  }
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // Holds the index where the neighbor was found for backprop.
  else if (top.size() == 1) {
    if (bottom.size() > 1) {
      index_mask_.Reshape(bottom[0]->num(), channels_, num_output_, bottom[1]->height());
    } else {
      index_mask_.Reshape(bottom[0]->num(), channels_, num_output_, 1);
    }
  }
}

template <typename Dtype>
void ResampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //TODO: Implement CPU Version
}

template <typename Dtype>
void ResampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //TODO: Implement CPU Version
}


#ifdef CPU_ONLY
STUB_GPU(ResampleLayer);
#endif

INSTANTIATE_CLASS(ResampleLayer);
REGISTER_LAYER_CLASS(Resample);

}  // namespace caffe
