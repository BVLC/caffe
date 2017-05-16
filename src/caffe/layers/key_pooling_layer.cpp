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
  key_top_mask_.Reshape(required_shape);
  if (top.size() > 1) {
    top[1]->Reshape(required_shape);
    key_top_mask_.ShareData(*top[1]);
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
    Blob<Dtype> per_key_bottom;
    Blob<Dtype> per_key_top;
    Blob<Dtype> per_key_top_mask;
    vector<Blob<Dtype>*> pooling_bottoms;
    vector<Blob<Dtype>*> pooling_tops;
    pooling_bottoms.push_back(&per_key_bottom);
    pooling_tops.push_back(&per_key_top);
    pooling_tops.push_back(&per_key_top_mask);

    vector<int> bottom_shape = bottom[0]->shape();
    vector<int> top_shape = top[0]->shape();
    bottom_shape[0] = key_len_[i];
    top_shape[0] = key_len_[i];
    per_key_bottom.Reshape(bottom_shape);
    per_key_top_mask.Reshape(top_shape);

#ifdef USE_CPU_INPLACE
    // Set the bottom as a view into the alocated blob.
    per_key_bottom.set_cpu_data(
        &bottom[0]->mutable_cpu_data()[bottom[0]->offset(key_start_[i])]);
    // Set the top mast as a view into the allocated blob.
    per_key_top_mask.set_cpu_data(
        &key_top_mask_.mutable_cpu_data()[key_top_mask_.offset(i)]);
#else
    caffe_copy(per_key_bottom.count(),
               &bottom[0]->cpu_data()[bottom[0]->offset(key_start_[i])],
               per_key_bottom.mutable_cpu_data());
    caffe_copy(per_key_top_mask.count(),
               &key_top_mask_.cpu_data()[key_top_mask_.offset(i)],
               per_key_top_mask.mutable_cpu_data());
#endif
    // Perform pooling on this key.
    pooling_layer_.Forward(pooling_bottoms, pooling_tops);

#ifdef USE_CPU_INPLACE
    // TODO: Currently the max pooling layer returns indices for each image
    // channel. Update these by adding the collection and channel offsets.
    Dtype* top_mask =
        &key_top_mask_.mutable_cpu_data()[key_top_mask_.offset(i)];
#else
    Dtype* top_mask = per_key_top_mask.mutable_cpu_data();
#endif
    for (int j = 0; j < per_key_top.shape(0); ++j) {
      for (int c = 0; c < per_key_top.shape(1); ++c) {
        const int global_offset = bottom[0]->offset(key_start_[i] + j, c);
        const int local_offset = per_key_top.offset(j, c);
        for (int k = 0; k < per_key_top.count(2); ++k) {
          top_mask[local_offset + k] += global_offset;
        }
      }
    }

    const Dtype *key_pool = per_key_top.cpu_data();
    Dtype *top_data = &top[0]->mutable_cpu_data()[top[0]->offset(i)];
    caffe_copy(top[0]->count(1), key_pool, top_data);

    for (int j = 1; j < per_key_top.shape(0); ++j) {
      const int j_offset = per_key_top.offset(j);
      for (int pi = 0; pi < per_key_top.count(1); ++pi) {
        if (key_pool[j_offset + pi] > top_data[pi]) {
          top_data[pi] = key_pool[j_offset + pi];
          top_mask[pi] = top_mask[j_offset + pi];
        }

      }
    }

#ifdef USE_CPU_INPLACE
#else
    caffe_copy(per_key_top_mask.count(),
           per_key_top_mask.cpu_data(),
           &key_top_mask_.mutable_cpu_data()[key_top_mask_.offset(i)]);
#endif

  }

  if (top.size() > 2) {
    caffe_copy(has_keys_.size(), &has_keys_[0], top[2]->mutable_cpu_data());
  }
}

template <typename Dtype>
void KeyPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {

  Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype *top_diff = top[0]->cpu_diff();
  const Dtype *mask = key_top_mask_.cpu_data();
  for (int i = 0; i < has_keys_.size(); ++i) {
    const int top_offset = top[0]->offset(i);
    for (int pi = 0; pi < key_top_mask_.count(1); ++pi) {
      const int bottom_index = mask[top_offset + pi];
      bottom_diff[bottom_index] += top_diff[top_offset + pi];
    }
  }
}

INSTANTIATE_CLASS(KeyPoolingLayer);

}  // namespace caffe
