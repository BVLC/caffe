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
  key_len_.clear();

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

    for (int i = 1; i < num_keys; ++i) {
      if (keys[i] != current_key) {
        key_len_.push_back(i - key_start_[key_start_.size()-1]);

        current_key = keys[i];
        has_keys_.push_back(current_key);
        key_start_.push_back(i);
      }
    }
    key_len_.push_back(num_keys - key_start_[key_start_.size()-1]);
  }

  CHECK_LE(has_keys_.size(), num_keys);
  CHECK_EQ(has_keys_.size(), key_start_.size());
  CHECK_EQ(has_keys_.size(), key_len_.size());

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
    // Create local blobs for the per-key pooling.
    Blob<Dtype> key_bottom;
    Blob<Dtype> key_top;
    vector<Blob<Dtype>*> pooling_bottoms;
    vector<Blob<Dtype>*> pooling_tops;
    pooling_bottoms.push_back(&key_bottom);
    pooling_tops.push_back(&key_top);

    vector<int> bottom_shape = bottom[0]->shape();
    bottom_shape[0] = key_len_[i];
    key_bottom.Reshape(bottom_shape);

    // Set the bottom as a view into the alocated blob.
    key_bottom.set_cpu_data(
        &bottom[0]->mutable_cpu_data()[bottom[0]->offset(key_start_[i])]);

    // Perform pooling on this key.
    pooling_layer_.Forward(pooling_bottoms, pooling_tops);

    const Dtype *key_pool = key_top.cpu_data();
    Dtype *top_data = &top[0]->mutable_cpu_data()[top[0]->offset(i)];
    caffe_copy(top[0]->count(1), key_pool, top_data);

    Dtype *top_mask = NULL;
    if (top.size() > 1) {
      top_mask = &top[1]->mutable_cpu_data()[top[1]->offset(i)];
      caffe_set(top[1]->count(1), Dtype(key_start_[i]), top_mask);
    }

    for (int j = 1; j < key_top.shape(0); ++j) {
      int j_offset = key_top.offset(j);
      for (int k = 0; k < key_top.count(1); ++k) {
        if (key_pool[j_offset + k] > top_data[k]) {
          top_data[k] = key_pool[j_offset + k];
          if (top_mask) {
            top_mask[k] = j + key_start_[i];
          }
        }
      }
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
