#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/key_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void KeyPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "The number of keys must be equal to the size of the batch";

  pooling_layer_.LayerSetUp(bottom, top);
}

template <typename Dtype>
void KeyPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  int num_keys = bottom[1]->shape(0);
  has_keys_.clear();
  key_start_.clear();
  key_end_.clear();

  vector<Blob<Dtype>*> pooling_top;
  pooling_top.push_back(top[0]);
  if (top.size() > 1) {
    pooling_top.push_back(top[1]);
  }

  pooling_layer_.Reshape(bottom, pooling_top);

  if (num_keys > 0) {
    const Dtype* keys = bottom[1]->cpu_data();

    Dtype current_key = keys[0];
    has_keys_.push_back(current_key);
    key_start_.push_back(0);

    int j = 0;
    for (int i = 1; i < num_keys; ++i) {
      if (keys[i] != current_key) {
        j++;
        key_end_.push_back(i);

        largest_key_set_ =
            std::max(key_end_[j - 1] - key_start_[j - 1], largest_key_set_);

        current_key = keys[i];
        has_keys_.push_back(current_key);
        key_start_.push_back(i);
      }
    }
    key_end_.push_back(num_keys);

    largest_key_set_ = std::max(
        key_end_[num_keys - 1] - key_start_[num_keys - 1], largest_key_set_);
  }

  CHECK_LE(has_keys_.size(), num_keys);

  // Resize the tops to match the keys.
  vector<int> required_shape(top[0]->shape());
  required_shape[0] = has_keys_.size();
  top[0]->Reshape(required_shape);
  if (top.size() > 1) {
    top[1]->Reshape(required_shape);
  }

  if (top.size() > 2) {
    top[2]->Reshape(has_keys_.size(), 1, 1, 1);
  }
}

template <typename Dtype>
void KeyPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < has_keys_.size(); ++i) {
    // Create a local copy of the blobs for the top and the bottom
    Blob<Dtype> key_bottom;
    Blob<Dtype> key_top;
    Blob<Dtype> key_top_mask;
    vector<Blob<Dtype>*> pooling_bottoms;
    vector<Blob<Dtype>*> pooling_tops;
    pooling_bottoms.push_back(&key_bottom);
    pooling_tops.push_back(&key_top);

    vector<int> bottom_shape = bottom[0]->shape();
    bottom_shape[0] = key_end_[i] - key_start_[i];
    key_bottom.Reshape(bottom_shape);

    const Dtype* bottom_data = bottom[0]->cpu_data();
    caffe_copy(key_bottom.count(),
               &bottom_data[bottom[0]->offset(key_start_[i])],
               key_bottom.mutable_cpu_data());

    // Perform pooling on this key.
    pooling_layer_.Forward(pooling_bottoms, pooling_tops);

    key_top_mask.Reshape(key_top.shape());
    Dtype* key_pool = key_top.mutable_cpu_data();
    Dtype* key_mask = key_top_mask.mutable_cpu_data();

    caffe_set(key_top_mask.count(), Dtype(key_start_[i]), key_mask);
    for (int j = 1; j < key_top.shape(0); ++j) {
      int j_offset = key_top.offset(j);
      for (int k = 0; k < key_top.count(1); ++k) {
        if (key_pool[j_offset + k] > key_pool[k]) {
          key_pool[k] = key_pool[j_offset + k];
          key_mask[k] = j + key_start_[i];
        }
      }
    }

    Dtype* top_data = top[0]->mutable_cpu_data();
    caffe_copy(key_top.count(1), key_top.cpu_data(),
               &top_data[top[0]->offset(i)]);
    if (top.size() > 1) {
      Dtype* top_mask = top[1]->mutable_cpu_data();
      caffe_copy(key_top_mask.count(1), key_top_mask.cpu_data(),
                 &top_mask[top[1]->offset(i)]);
    }
  }

  if (top.size() > 2) {
    caffe_copy(has_keys_.size(), &has_keys_[0], top[2]->mutable_cpu_data());
  }
}

template <typename Dtype>
void KeyPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {
  pooling_layer_.Backward(bottom, propagate_down, top);
}

INSTANTIATE_CLASS(KeyPoolingLayer);

}  // namespace caffe
