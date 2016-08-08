#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/recurrent_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RecurrentLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Hacky fix for test time... reshare all the shared blobs.
  // TODO: somehow make this work non-hackily.
  if (this->phase_ == TEST) {
    unrolled_net_->ShareWeights();
  }

  DCHECK_EQ(recur_input_blobs_.size(), recur_output_blobs_.size());
  if (!expose_hidden_) {
    for (int i = 0; i < recur_input_blobs_.size(); ++i) {
      const int count = recur_input_blobs_[i]->count();
      DCHECK_EQ(count, recur_output_blobs_[i]->count());
      const Dtype* timestep_T_data = recur_output_blobs_[i]->gpu_data();
      Dtype* timestep_0_data = recur_input_blobs_[i]->mutable_gpu_data();
      caffe_copy(count, timestep_T_data, timestep_0_data);
    }
  }

  unrolled_net_->ForwardTo(last_layer_index_);

  if (expose_hidden_) {
    const int top_offset = output_blobs_.size();
    for (int i = top_offset, j = 0; i < top.size(); ++i, ++j) {
      top[i]->ShareData(*recur_output_blobs_[j]);
    }
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(RecurrentLayer);

}  // namespace caffe
