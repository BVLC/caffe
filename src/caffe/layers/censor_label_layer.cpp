#include <vector>
#include <set>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::set;

template <typename Dtype>
void CensorLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CensorLabelParameter param = this->layer_param_.censor_label_param();
  ignore_label_ = param.ignore_label();
}

template <typename Dtype>
void CensorLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(channels_, 1) << "bottom[0] must have one channel";
  CHECK_EQ(bottom[1]->num(), num_);
  max_labels_ = bottom[1]->channels();
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  top[0]->Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void CensorLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // We use the value -1 as the filler to make sure the label list has max_labels members
  caffe_set(top[0]->count(), Dtype(-1), top[0]->mutable_cpu_data());
  for (int n = 0; n < num_; ++n) {
    set<Dtype> vals;
    const Dtype *bottom1_data = bottom[1]->cpu_data(n);
    for (int i = 0; i < max_labels_; ++i) {
      if (bottom1_data[i] != -1) {
	vals.insert(bottom1_data[i]);
      }
    }
    const Dtype *bottom_data = bottom[0]->cpu_data(n, 0);
    Dtype *top_data = top[0]->mutable_cpu_data(n);
    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < width_; ++w) {
	top_data[h * width_ + w] = (vals.count(bottom_data[h * width_ + w]) != 0) ?
	  bottom_data[h * width_ + w] : ignore_label_;
      }
    }
  }
}

template <typename Dtype>
void CensorLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // Does nothing
}

#ifdef CPU_ONLY
//STUB_GPU(CensorLabelLayer);
#endif

INSTANTIATE_CLASS(CensorLabelLayer);
REGISTER_LAYER_CLASS(CENSOR_LABEL, CensorLabelLayer);

}  // namespace caffe
