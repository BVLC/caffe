#include <vector>

#include "caffe/layers/permute_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Permute(const int count, Dtype *bottom_data, const int *permute_order,
             const int *old_steps, const int *new_steps, const int num_axes,
             Dtype *top_data) {
  for (int i = 0; i < count; ++i) {
    int old_idx = 0;
    int idx = i;
    for (int j = 0; j < num_axes; ++j) {
      int order = permute_order[j];
      old_idx += (idx / new_steps[j]) * old_steps[order];
      idx %= new_steps[j];
    }
    top_data[i] = bottom_data[old_idx];
  }
}

template <typename Dtype>
void PermuteLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  PermuteParameter permute_param = this->layer_param_.permute_param();
  CHECK_EQ(bottom.size(), 1);
  num_axes_ = bottom[0]->num_axes();
  vector<int> orders;
  // Push the specified new orders.
  for (int i = 0; i < permute_param.order_size(); ++i) {
    int order = permute_param.order(i);
    CHECK_LT(order, num_axes_)
        << "order should be less than the input dimension.";
    if (std::find(orders.begin(), orders.end(), order) != orders.end()) {
      LOG(FATAL) << "there are duplicate orders";
    }
    orders.push_back(order);
  }
  // Push the rest orders. And save original step sizes for each axis.
  for (int i = 0; i < num_axes_; ++i) {
    if (std::find(orders.begin(), orders.end(), i) == orders.end()) {
      orders.push_back(i);
    }
  }
  CHECK_EQ(num_axes_, orders.size());
  // Check if we need to reorder the data or keep it.
  need_permute_ = false;
  for (int i = 0; i < num_axes_; ++i) {
    if (orders[i] != i) {
      // As long as there is one order which is different from the natural order
      // of the data, we need to permute. Otherwise, we share the data and diff.
      need_permute_ = true;
      break;
    }
  }

  permute_order_.Reshape(num_axes_, 1, 1, 1);
  vector<int> top_shape(num_axes_, 1);
  for (int i = 0; i < num_axes_; ++i) {
    permute_order_.mutable_cpu_data()[i] = orders[i];
    top_shape[i] = bottom[0]->shape(orders[i]);
  }
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void PermuteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  Forward_const_cpu(bottom, top);
}

template <typename Dtype>
void PermuteLayer<Dtype>::Forward_const_cpu(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) const {
  if (need_permute_) {
    vector<int> top_shape;
    Blob<int> old_steps;
    Blob<int> new_steps;
    old_steps.Reshape(num_axes_, 1, 1, 1);
    new_steps.Reshape(num_axes_, 1, 1, 1);
    for (int i = 0; i < num_axes_; ++i) {
      if (i == num_axes_ - 1) {
        old_steps.mutable_cpu_data()[i] = 1;
      } else {
        old_steps.mutable_cpu_data()[i] = bottom[0]->count(i + 1);
      }
      top_shape.push_back(bottom[0]->shape(permute_order_.cpu_data()[i]));
    }
    top[0]->Reshape(top_shape);

    for (int i = 0; i < num_axes_; ++i) {
      if (i == num_axes_ - 1) {
        new_steps.mutable_cpu_data()[i] = 1;
      } else {
        new_steps.mutable_cpu_data()[i] = top[0]->count(i + 1);
      }
    }

    Dtype *bottom_data = bottom[0]->mutable_cpu_data();
    Dtype *top_data = top[0]->mutable_cpu_data();
    const int top_count = top[0]->count();
    const int *permute_order = permute_order_.cpu_data();
    Permute(top_count, bottom_data, permute_order, old_steps.cpu_data(),
            new_steps.cpu_data(), num_axes_, top_data);
  } else {
    // If there is no need to permute, we share data to save memory.
    top[0]->ShareData(*bottom[0]);
  }
}

#ifdef CPU_ONLY
STUB_GPU(PermuteLayer);
STUB_GPU_FORWARD_CONST(PermuteLayer, Forward_const);
#endif

INSTANTIATE_CLASS(PermuteLayer);
REGISTER_LAYER_CLASS(Permute);

} // namespace caffe
