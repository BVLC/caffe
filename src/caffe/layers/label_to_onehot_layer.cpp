#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/label_to_onehot_layer.hpp"

namespace caffe {

template <typename Dtype>
void LabelToOnehotLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void LabelToOnehotLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), this->layer_param_.inner_product_param().num_output(), 1, 1);
}

template <typename Dtype>
void LabelToOnehotLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->num();
  const int count = top[0]->count();
  const int dim = count / num;
  int label;
  caffe_set(count, Dtype(0), top_data);
  for (int i = 0; i < num; ++i) {
    label = static_cast<int>(bottom_data[i]);
    if (label >=0 && label < dim)
      top_data[i * dim + label] = 1.;
    else
      LOG(FATAL) << "Label " << label << " outside of expected range [0," << dim-1 << "]";
  }
}

template <typename Dtype>
void LabelToOnehotLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "LabelToOnehot layer does not do backward";
}


#ifdef CPU_ONLY
STUB_GPU(LabelToOnehotLayer);
#endif

INSTANTIATE_CLASS(LabelToOnehotLayer);
REGISTER_LAYER_CLASS(LabelToOnehot);

}  // namespace caffe

