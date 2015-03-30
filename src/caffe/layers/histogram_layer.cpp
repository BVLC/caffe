#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void HistogramLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  HistogramParameter param = this->layer_param_.histogram_param();
  num_labels_ = param.num_labels();
  CHECK_GT(num_labels_, 0) << "num_labels needs to be positive";
  for (int i = 0; i < param.ignore_label_size(); ++i){
    ignore_label_.insert(param.ignore_label(i));
  }
}

template <typename Dtype>
void HistogramLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  CHECK_EQ(bottom[0]->channels(), 1) << "Input must have single channel";
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  //
  top[0]->Reshape(num_, num_labels_, 1, 1);
}

template <typename Dtype>
void HistogramLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  const int spatial_dim = height_ * width_;
  for (int n = 0; n < num_; ++n) {
    const Dtype* bottom_data = bottom[0]->cpu_data(n);
    Dtype* top_data = top[0]->mutable_cpu_data();
    for (int j = 0; j < spatial_dim; ++j) {
      const int label = static_cast<int>(bottom_data[j]);
      CHECK_EQ(bottom_data[j], label) << "Input assumed integer-valued";
      if (ignore_label_.count(label) != 0) {
	continue;
      } else if (label >= 0 && label < num_labels_) {
	// Increment the count bin
	top_data[label] += Dtype(1);
      } else {
	LOG(FATAL) << "Unexpected label " << label;
      }
    }
  }
}

#ifdef CPU_ONLY
//STUB_GPU(HistogramLayer);
#endif

INSTANTIATE_CLASS(HistogramLayer);

}  // namespace caffe
