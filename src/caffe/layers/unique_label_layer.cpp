#include <vector>
#include <set>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::set;

template <typename Dtype>
void UniqueLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  UniqueLabelParameter param = this->layer_param_.unique_label_param();
  for (int i = 0; i < param.ignore_label_size(); ++i){
    ignore_label_.insert(param.ignore_label(i));
  }
  for (int i = 0; i < param.force_label_size(); ++i){
    force_label_.insert(param.force_label(i));
  }
  max_labels_ = param.max_labels();
  CHECK_GT(max_labels_, force_label_.size()) << 
    "At least one label more than the forced ones needs to fit";
}

template <typename Dtype>
void UniqueLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(channels_, 1) << "Input must have one channel";
  top[0]->Reshape(num_, max_labels_, 1, 1);
}

template <typename Dtype>
void UniqueLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // We use the value -1 as the filler to make sure the label list has max_labels members
  caffe_set(top[0]->count(), Dtype(-1), top[0]->mutable_cpu_data());
  for (int n = 0; n < num_; ++n) {
    set<Dtype> vals;
    const Dtype *bottom_data = bottom[0]->cpu_data(n);
    // add the vals present in the image
    for (int i = 0; i < height_ * width_; ++i) {
      if (ignore_label_.count(bottom_data[i]) == 0) {
	vals.insert(bottom_data[i]);
      }
    }
    // add the vals in the forced label set
    for (typename set<Dtype>::iterator it = force_label_.begin(); it != force_label_.end(); ++it) {
      vals.insert(*it);
    }
    CHECK_LE(vals.size(), max_labels_) << "Too many unique elements, increase capacity of UniqueLabelLayer";
    Dtype *top_data = top[0]->mutable_cpu_data(n);
    int j = 0;
    for (typename set<Dtype>::iterator it = vals.begin(); it != vals.end(); ++it) {
      top_data[j++] = *it;
    }
  }
}

template <typename Dtype>
void UniqueLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    LOG(FATAL) << "Cannot propagate down to label input";
  }
}

#ifdef CPU_ONLY
//STUB_GPU(UniqueLabelLayer);
#endif

INSTANTIATE_CLASS(UniqueLabelLayer);
REGISTER_LAYER_CLASS(UNIQUE_LABEL, UniqueLabelLayer);

}  // namespace caffe
