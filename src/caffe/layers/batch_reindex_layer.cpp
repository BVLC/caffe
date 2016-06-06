#include <vector>

#include "caffe/layers/batch_reindex_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void BatchReindexLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(1, bottom[1]->num_axes());
  vector<int> newshape;
  newshape.push_back(bottom[1]->shape(0));
  for (int i = 1; i < bottom[0]->shape().size(); ++i) {
    newshape.push_back(bottom[0]->shape()[i]);
  }
  top[0]->Reshape(newshape);
}

template<typename Dtype>
void BatchReindexLayer<Dtype>::check_batch_reindex(int initial_num,
                                                   int final_num,
                                                   const Dtype* ridx_data) {
  for (int i = 0; i < final_num; ++i) {
    CHECK_GE(ridx_data[i], 0)
        << "Index specified for reindex layer was negative.";
    CHECK_LT(ridx_data[i], initial_num)
        << "Index specified for reindex layer was greater than batch size.";
  }
}

template<typename Dtype>
void BatchReindexLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  check_batch_reindex(bottom[0]->shape(0), bottom[1]->count(),
                      bottom[1]->cpu_data());
  if (top[0]->count() == 0) {
    return;
  }
  int inner_dim = bottom[0]->count() / bottom[0]->shape(0);
  const Dtype* in = bottom[0]->cpu_data();
  const Dtype* permut = bottom[1]->cpu_data();
  Dtype* out = top[0]->mutable_cpu_data();
  for (int index = 0; index < top[0]->count(); ++index) {
    int n = index / (inner_dim);
    int in_n = static_cast<int>(permut[n]);
    out[index] = in[in_n * (inner_dim) + index % (inner_dim)];
  }
}

template<typename Dtype>
void BatchReindexLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[1]) << "Cannot backprop to index.";
  if (!propagate_down[0]) {
    return;
  }
  int inner_dim = bottom[0]->count() / bottom[0]->shape(0);
  Dtype* bot_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* permut = bottom[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bot_diff);
  for (int index = 0; index < top[0]->count(); ++index) {
    int n = index / (inner_dim);
    int in_n = static_cast<int>(permut[n]);
    bot_diff[in_n * (inner_dim) + index % (inner_dim)] += top_diff[index];
  }
}

#ifdef CPU_ONLY
STUB_GPU(BatchReindexLayer);
#endif

INSTANTIATE_CLASS(BatchReindexLayer);
REGISTER_LAYER_CLASS(BatchReindex);

}  // namespace caffe
