#include <vector>

#include "caffe/layers/split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void SplitLayer<Dtype, MItype, MOtype>::Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  count_ = bottom[0]->count();
  for (int_tp i = 0; i < top.size(); ++i) {
    // Do not allow in-place computation in the SplitLayer.  Instead, share data
    // by reference in the forward pass, and keep separate diff allocations in
    // the backward pass.  (Technically, it should be possible to share the diff
    // blob of the first split output with the input, but this seems to cause
    // some strange effects in practice...)
    CHECK_NE(top[i], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
    top[i]->ReshapeLike(*bottom[0]);
    CHECK_EQ(count_, top[i]->count());
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void SplitLayer<Dtype, MItype, MOtype>::Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  for (int_tp i = 0; i < top.size(); ++i) {
    top[i]->ShareData(*bottom[0]);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void SplitLayer<Dtype, MItype, MOtype>::Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<MItype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  if (top.size() == 1) {
    caffe_copy(count_, top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
    return;
  }
  caffe_add(count_, top[0]->cpu_diff(), top[1]->cpu_diff(),
            bottom[0]->mutable_cpu_diff());
  // Add remaining top blob diffs.
  for (int_tp i = 2; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_axpy(count_, Dtype(1.), top_diff, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(SplitLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(SplitLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(SplitLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(SplitLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(Split);
REGISTER_LAYER_CLASS_INST(Split, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(Split, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(Split, (double), (double), (double));

}  // namespace caffe
