#include <vector>

#include "caffe/layers/reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void ReductionLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                          const vector<Blob<MItype>*>& bottom,
                                          const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<const Dtype> mult_data;

  int_tp bottom_data_off = 0;
  int_tp top_data_off = 0;

  if (sum_multiplier_.count() > 0) {
    mult_data = sum_multiplier_.gpu_data();
  }
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int_tp i = 0; i < num_; ++i) {
    switch (op_) {
      case ReductionParameter_ReductionOp_SUM:
      case ReductionParameter_ReductionOp_MEAN:
        this->device_->template dot<Dtype>(dim_, mult_data,
                                           bottom_data + bottom_data_off,
                                           top_data + top_data_off);
        break;
      case ReductionParameter_ReductionOp_ASUM:
        this->device_->template asum<Dtype>(dim_, bottom_data + bottom_data_off,
                                            top_data + top_data_off);
        break;
      case ReductionParameter_ReductionOp_SUMSQ:
        this->device_->template dot<Dtype>(dim_, bottom_data + bottom_data_off,
                                           bottom_data + bottom_data_off,
                                           top_data + top_data_off);
        break;
      default:
        LOG(FATAL)<< "Unknown reduction op: "
        << ReductionParameter_ReductionOp_Name(op_);
      }
      bottom_data_off += dim_;
      ++top_data_off;
    }
  if (coeff_ != Dtype(1)) {
    // Reset the top_data pointer.
    vptr<Dtype> top_gpu_data = top[0]->mutable_gpu_data();
    this->device_->scal(num_, coeff_, top_gpu_data);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void ReductionLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                          const vector<Blob<MOtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<MItype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  // Get bottom_data, if needed.
  vptr<const Dtype> bottom_data;
  switch (op_) {
    // Operations that don't need bottom_data
    case ReductionParameter_ReductionOp_SUM:
    case ReductionParameter_ReductionOp_MEAN:
      break;
      // Operations that need bottom_data
    case ReductionParameter_ReductionOp_ASUM:
    case ReductionParameter_ReductionOp_SUMSQ:
      bottom_data = bottom[0]->gpu_data();
      break;
    default:
      LOG(FATAL)<< "Unknown reduction op: "
      << ReductionParameter_ReductionOp_Name(op_);
    }
  const Dtype* top_diff = top[0]->cpu_diff();
  vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();

  int_tp bottom_data_off = 0;
  int_tp bottom_diff_off = 0;
  int_tp top_diff_off = 0;

  for (int_tp i = 0; i < num_; ++i) {
    const Dtype bottom_coeff = (*(top_diff + top_diff_off)) * coeff_;
    switch (op_) {
      case ReductionParameter_ReductionOp_SUM:
      case ReductionParameter_ReductionOp_MEAN:
        this->device_->set(dim_, bottom_coeff, bottom_diff + bottom_diff_off);
        break;
      case ReductionParameter_ReductionOp_ASUM:
        this->device_->template sign<Dtype>(dim_, bottom_data + bottom_data_off,
                                            bottom_diff + bottom_diff_off);
        this->device_->template scal<Dtype>(dim_, bottom_coeff,
                                            bottom_diff + bottom_diff_off);
        break;
      case ReductionParameter_ReductionOp_SUMSQ:
        this->device_->template scale<Dtype>(dim_, 2 * bottom_coeff,
                                             bottom_data + bottom_data_off,
                                             bottom_diff + bottom_diff_off);
        break;
      default:
        LOG(FATAL)<< "Unknown reduction op: "
        << ReductionParameter_ReductionOp_Name(op_);
      }
    bottom_data_off += dim_;
    bottom_diff_off += dim_;
    ++top_diff_off;
  }
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReductionLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReductionLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReductionLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReductionLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReductionLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReductionLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
