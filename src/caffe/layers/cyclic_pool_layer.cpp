#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/cyclic_pool_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void CyclicPoolLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
  CHECK_EQ(bottom[0]->height(), bottom[0]->width()) <<
    "feature maps must be square";
  CHECK_EQ(bottom[0]->num()%4, 0) <<
    "number of batches must can be divided by 4";
}

template <typename Dtype>
void CyclicPoolLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape = bottom[0]->shape();
  shape[0] /= 4;
  top[0]->Reshape(shape);
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX) {
    max_idx_.Reshape(top[0]->shape());
  }
}

template <typename Dtype>
void CyclicPoolLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->shape(0);
  const int batch_dim = bottom[0]->count(1);
  const int top_count = top[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int* mask = max_idx_.mutable_cpu_data();
  // different pooling method
  switch (this->layer_param_.cyclic_pool_param().pool()) {
    case CyclicPoolParameter_PoolMethod_AVE:
    caffe_set(top_count, Dtype(0), top_data);
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < batch_dim; ++j) {
        top_data[i/4*batch_dim+j] += bottom_data[i*batch_dim+j]/4.;
      }
    }
    break;
    case CyclicPoolParameter_PoolMethod_MAX:
    caffe_set(top_count, -1, mask);
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    int bottom_index, top_index;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < batch_dim; ++j) {
        bottom_index = i * batch_dim + j;
        top_index = i/4 * batch_dim + j;
        if (top_data[top_index] < bottom_data[bottom_index]) {
          top_data[top_index] =  bottom_data[bottom_index];
          mask[top_index] = bottom_index;
        }
      }
    }
    break;
    case CyclicPoolParameter_PoolMethod_RMS:
    CHECK(0) << "currently not supported";
    break;
    default:
    break;
  }
}

template <typename Dtype>
void CyclicPoolLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const int num = bottom[0]->shape(0);
  const int batch_dim = top[0]->count(1);
  const int top_count = top[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int* mask = max_idx_.cpu_data();
  // clear bottom diff
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // different pooling method
  switch (this->layer_param_.cyclic_pool_param().pool()) {
    case CyclicPoolParameter_PoolMethod_AVE:
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < batch_dim; ++j) {
        bottom_diff[i*batch_dim+j] += top_diff[i/4*batch_dim+j]/Dtype(4);
      }
    }
    break;
    case CyclicPoolParameter_PoolMethod_MAX:
    for (int i = 0; i < top_count; ++i) {
      bottom_diff[mask[i]] += top_diff[i];
    }
    break;
    case CyclicPoolParameter_PoolMethod_RMS:
    CHECK(0) << "currently not supported";
    break;
    default:
    break;
  }
}

#ifdef CPU_ONLY
STUB_GPU(CyclicPoolLayer);
#endif

INSTANTIATE_CLASS(CyclicPoolLayer);
REGISTER_LAYER_CLASS(CyclicPool);

}  // namespace caffe
