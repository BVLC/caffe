#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void FlattenLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  int channels_out = bottom[0]->channels() * bottom[0]->height()
      * bottom[0]->width();
  (*top)[0]->Reshape(bottom[0]->num(), channels_out, 1, 1);
  count_ = bottom[0]->num() * channels_out;
  CHECK_EQ(count_, bottom[0]->count());
  CHECK_EQ(count_, (*top)[0]->count());
}

template <typename Dtype>
void FlattenLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void FlattenLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  (*bottom)[0]->ShareDiff(*top[0]);
}

#ifdef CPU_ONLY
STUB_GPU(FlattenLayer);
#endif

INSTANTIATE_CLASS(FlattenLayer);

}  // namespace caffe
