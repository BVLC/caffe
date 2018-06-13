#include <vector>

#include "caffe/layers/split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void SplitLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                      const vector<Blob<MItype>*>& bottom,
                                      const vector<Blob<MOtype>*>& top) {
  for (int_tp i = 0; i < top.size(); ++i) {
    if (this->layer_param().bottom_shared_index_size() > 0 ||
        this->layer_param().top_shared_index_size() > 0) {
      // Using shared blobs, copy data instead of sharing the blob
      this->device_->template copy<Dtype>(bottom[0]->count(),
                                          bottom[0]->gpu_data(),
                                          top[i]->mutable_gpu_data());
    } else {
    // Normal mode
      top[i]->ShareData(*bottom[0]);
    }
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void SplitLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                      const vector<Blob<MOtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<MItype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  if (top.size() == 1) {
    this->device_->template copy<Dtype>(count_, top[0]->gpu_diff(),
                                        bottom[0]->mutable_gpu_diff());
    return;
  }
  this->device_->template add<Dtype>(count_, top[0]->gpu_diff(),
                 top[1]->gpu_diff(), bottom[0]->mutable_gpu_diff());
  // Add remaining top blob diffs.
  for (int_tp i = 2; i < top.size(); ++i) {
    vptr<const Dtype> top_diff = top[i]->gpu_diff();
    vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
    this->device_->template axpy<Dtype>(count_, Dtype(1.), top_diff,
                                        bottom_diff);
  }
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(SplitLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SplitLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SplitLayer, Forward_gpu,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SplitLayer, Forward_gpu,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SplitLayer, Forward_gpu,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SplitLayer, Forward_gpu,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SplitLayer, Forward_gpu,
                                  (uint64_t), (uint64_t), (uint64_t));


INSTANTIATE_CLASST_FUNC_3T_GUARDED(SplitLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SplitLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SplitLayer, Backward_gpu,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SplitLayer, Backward_gpu,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SplitLayer, Backward_gpu,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SplitLayer, Backward_gpu,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SplitLayer, Backward_gpu,
                                  (uint64_t), (uint64_t), (uint64_t));

}  // namespace caffe
