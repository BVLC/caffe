#include <algorithm>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void NumpyDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void NumpyDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape;
  const int shape_size = this->runtime_param().numpy_data_param().shape_size();
  for (int i = 0; i < shape_size; ++i) {
    shape.push_back(this->runtime_param().numpy_data_param().shape(i));
    ASSERT(shape[i] > 0, "All numpy data dimensions must be non-zero");
  }
  top[0]->Reshape(shape);
}

template <typename Dtype>
void NumpyDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  std::copy(this->runtime_param().numpy_data_param().data().begin(),
            this->runtime_param().numpy_data_param().data().end(),
            top_data);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(NumpyDataLayer, Forward);
#endif

INSTANTIATE_CLASS(NumpyDataLayer);
REGISTER_LAYER_CLASS(NumpyData);

}  // namespace caffe
